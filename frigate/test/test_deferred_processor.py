"""Tests for DeferredRealtimeProcessorApi."""

import sys
import time
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from pydantic import ValidationError

from frigate.config.classification import (
    CustomClassificationConfig,
    CustomClassificationObjectConfig,
)
from frigate.data_processing.real_time.api import DeferredRealtimeProcessorApi

# Mock TFLite before importing classification module
_MOCK_MODULES = [
    "tflite_runtime",
    "tflite_runtime.interpreter",
    "ai_edge_litert",
    "ai_edge_litert.interpreter",
]
for mod in _MOCK_MODULES:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from frigate.data_processing.real_time.custom_classification import (  # noqa: E402
    CustomObjectClassificationProcessor,
)


class StubDeferredProcessor(DeferredRealtimeProcessorApi):
    """Minimal concrete subclass for testing the deferred base."""

    def __init__(self, max_queue: int = 8):
        config = MagicMock()
        metrics = MagicMock()
        super().__init__(config, metrics, max_queue=max_queue)
        self.processed_items: list[tuple] = []

    def process_frame(self, obj_data: dict[str, Any], frame: np.ndarray) -> None:
        """Enqueue every call — no gating logic in the stub."""
        self._enqueue_task(("frame", obj_data, frame.copy()))

    def _process_task(self, task: tuple) -> None:
        kind = task[0]
        if kind == "frame":
            _, obj_data, frame = task
            self.processed_items.append((obj_data["id"], frame.shape))
            self._emit_result(
                {
                    "type": "test_result",
                    "id": obj_data["id"],
                    "label": "cat",
                    "score": 0.95,
                }
            )
        elif kind == "expire":
            _, object_id = task
            self.processed_items.append(("expired", object_id))

    def handle_request(
        self, topic: str, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        if topic == "reload":

            def _do_reload(data):
                return {"success": True, "model": data.get("name")}

            return self._enqueue_request(_do_reload, request_data)
        return None

    def expire_object(self, object_id: str, camera: str) -> None:
        self._enqueue_task(("expire", object_id))


class TestDeferredProcessorBase(unittest.TestCase):
    def test_enqueue_and_drain(self):
        """Tasks enqueued on main thread are processed by worker, results are drainable."""
        proc = StubDeferredProcessor()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        proc.process_frame({"id": "obj1"}, frame)
        proc.process_frame({"id": "obj2"}, frame)

        # Give the worker time to process
        time.sleep(0.1)

        results = proc.drain_results()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "obj1")
        self.assertEqual(results[1]["id"], "obj2")

        # Second drain should be empty
        self.assertEqual(len(proc.drain_results()), 0)

    def test_backpressure_drops_tasks(self):
        """When queue is full, new tasks are silently dropped."""
        proc = StubDeferredProcessor(max_queue=2)

        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        for i in range(10):
            proc.process_frame({"id": f"obj{i}"}, frame)

        time.sleep(0.2)
        results = proc.drain_results()
        # The key property: no crash, no unbounded growth
        self.assertLessEqual(len(results), 10)
        self.assertGreater(len(results), 0)

    def test_handle_request_through_worker(self):
        """handle_request blocks until the worker processes it and returns a response."""
        proc = StubDeferredProcessor()
        result = proc.handle_request("reload", {"name": "my_model"})
        self.assertEqual(result, {"success": True, "model": "my_model"})

    def test_expire_object_serialized_with_work(self):
        """expire_object goes through the queue, serialized with inference work."""
        proc = StubDeferredProcessor()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        proc.process_frame({"id": "obj1"}, frame)
        proc.expire_object("obj1", "front_door")

        time.sleep(0.1)
        # Both should have been processed in order
        self.assertEqual(len(proc.processed_items), 2)
        self.assertEqual(proc.processed_items[0][0], "obj1")
        self.assertEqual(proc.processed_items[1], ("expired", "obj1"))

    def test_shutdown_joins_worker(self):
        """shutdown() signals the worker to stop and joins the thread."""
        proc = StubDeferredProcessor()
        proc.shutdown()
        self.assertFalse(proc._worker.is_alive())

    def test_drain_results_returns_list(self):
        """drain_results returns a plain list, not a deque."""
        proc = StubDeferredProcessor()
        results = proc.drain_results()
        self.assertIsInstance(results, list)


class TestCustomObjectClassificationDeferred(unittest.TestCase):
    """Test that CustomObjectClassificationProcessor uses the deferred pattern correctly."""

    def _make_processor(self):
        config = MagicMock()
        model_config = MagicMock()
        model_config.name = "test_breed"
        model_config.object_config = MagicMock()
        model_config.object_config.objects = ["dog"]
        model_config.threshold = 0.5
        model_config.save_attempts = 10
        model_config.object_config.classification_type = "sub_label"
        model_config.object_config.min_area = None
        model_config.object_config.max_area = None
        publisher = MagicMock()
        requestor = MagicMock()
        metrics = MagicMock()
        metrics.classification_speeds = {}
        metrics.classification_cps = {}

        with patch.object(
            CustomObjectClassificationProcessor,
            "_CustomObjectClassificationProcessor__build_detector",
        ):
            proc = CustomObjectClassificationProcessor(
                config, model_config, publisher, requestor, metrics
            )
        proc.interpreter = None
        proc.tensor_input_details = [{"index": 0}]
        proc.tensor_output_details = [{"index": 0}]
        proc.labelmap = {0: "labrador", 1: "poodle", 2: "none"}
        return proc

    def test_is_deferred_processor(self):
        """CustomObjectClassificationProcessor should be a DeferredRealtimeProcessorApi."""
        proc = self._make_processor()
        self.assertIsInstance(proc, DeferredRealtimeProcessorApi)

    def test_expire_clears_history(self):
        """expire_object should clear classification history for the object."""
        proc = self._make_processor()
        proc.classification_history["obj1"] = [("labrador", 0.9, 1.0)]

        proc.expire_object("obj1", "front")
        time.sleep(0.1)

        self.assertNotIn("obj1", proc.classification_history)

    def test_drain_results_empty_when_no_model(self):
        """With no interpreter, process_frame saves training images but emits no results."""
        proc = self._make_processor()
        proc.interpreter = None

        frame = np.zeros((150, 100), dtype=np.uint8)
        obj_data = {
            "id": "obj1",
            "label": "dog",
            "false_positive": False,
            "end_time": None,
            "box": [10, 10, 50, 50],
            "camera": "front",
        }

        with patch(
            "frigate.data_processing.real_time.custom_classification.write_classification_attempt"
        ):
            proc.process_frame(obj_data, frame)

        time.sleep(0.1)
        results = proc.drain_results()
        self.assertEqual(len(results), 0)


class TestCustomObjectClassificationArea(unittest.TestCase):
    """Test min_area / max_area gating for object classification."""

    def _make_processor(
        self,
        min_area: int | float | None = None,
        max_area: int | float | None = None,
    ):
        config = MagicMock()
        config.cameras = {"front": MagicMock()}
        config.cameras["front"].detect.width = 100
        config.cameras["front"].detect.height = 100
        model_config = CustomClassificationConfig(
            name="test_breed",
            object_config=CustomClassificationObjectConfig(
                objects=["dog"], min_area=min_area, max_area=max_area
            ),
        )
        metrics = MagicMock()
        metrics.classification_speeds = {}
        metrics.classification_cps = {}

        with patch.object(
            CustomObjectClassificationProcessor,
            "_CustomObjectClassificationProcessor__build_detector",
        ):
            proc = CustomObjectClassificationProcessor(
                config, model_config, MagicMock(), MagicMock(), metrics
            )
        proc._enqueue_task = MagicMock()
        return proc

    def _process(self, proc, box: list[int]) -> bool:
        frame = np.zeros((150, 100), dtype=np.uint8)
        obj_data = {
            "id": "obj1",
            "label": "dog",
            "false_positive": False,
            "end_time": None,
            "box": box,
            "camera": "front",
        }
        proc.process_frame(obj_data, frame)
        return proc._enqueue_task.called

    def test_no_area_limits_classifies(self):
        proc = self._make_processor()
        self.assertTrue(self._process(proc, [10, 10, 15, 15]))

    def test_below_min_area_pixels_skipped(self):
        proc = self._make_processor(min_area=400)
        self.assertFalse(self._process(proc, [10, 10, 25, 25]))
        self.assertTrue(self._process(proc, [10, 10, 30, 30]))

    def test_above_max_area_pixels_skipped(self):
        proc = self._make_processor(max_area=400)
        self.assertFalse(self._process(proc, [10, 10, 40, 40]))
        self.assertTrue(self._process(proc, [10, 10, 30, 30]))

    def test_min_area_percentage(self):
        # 5% of a 100x100 detect frame is 500 pixels
        proc = self._make_processor(min_area=0.05)
        self.assertFalse(self._process(proc, [10, 10, 30, 30]))
        self.assertTrue(self._process(proc, [10, 10, 40, 40]))

    def test_skipped_object_does_not_add_history(self):
        proc = self._make_processor(min_area=400)
        self._process(proc, [10, 10, 15, 15])
        self.assertNotIn("obj1", proc.classification_history)


class TestCustomClassificationObjectConfigArea(unittest.TestCase):
    def test_defaults_are_none(self):
        cfg = CustomClassificationObjectConfig(objects=["dog"])
        self.assertIsNone(cfg.min_area)
        self.assertIsNone(cfg.max_area)

    def test_types_preserved(self):
        cfg = CustomClassificationObjectConfig(
            objects=["dog"], min_area=1000, max_area=0.5
        )
        self.assertEqual(cfg.min_area, 1000)
        self.assertIsInstance(cfg.min_area, int)
        self.assertEqual(cfg.max_area, 0.5)

    def test_invalid_percentage_rejected(self):
        with self.assertRaises(ValidationError):
            CustomClassificationObjectConfig(objects=["dog"], min_area=1.5)

    def test_negative_pixels_rejected(self):
        with self.assertRaises(ValidationError):
            CustomClassificationObjectConfig(objects=["dog"], min_area=-1)

    def test_min_greater_than_max_rejected(self):
        with self.assertRaises(ValidationError):
            CustomClassificationObjectConfig(
                objects=["dog"], min_area=5000, max_area=1000
            )


if __name__ == "__main__":
    unittest.main()
