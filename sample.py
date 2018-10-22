#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import json
import create_graph


# 推論部からresultを受け取る場合は必要なし
with open('test_data.txt', 'r') as f:
	test_data = json.load(f)
######################################

create_graph.create_graph(test_data)
