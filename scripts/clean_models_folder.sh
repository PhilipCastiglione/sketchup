#!/usr/bin/env bash
rm -rf \
  models/model/checkpoint \
  models/model/saved_model \
  models/model/eval_0/ \
  models/model/events.out.tfevents.* \
  models/model/export/ \
  models/model/graph.pbtxt \
  models/model/model.ckpt* \
  models/model/frozen_inference_graph.pb \
  models/model/pipeline.config

