#!/bin/bash
# redial: 0.5, 0.7, 0.9, 1, 3
dataset="redial"

nhop=1
drop_rate=0.5
. scripts/quick-start.sh

nhop=1
drop_rate=0.7
. scripts/quick-start.sh

nhop=1
drop_rate=0.9
. scripts/quick-start.sh

nhop=1
drop_rate=1.0
. scripts/quick-start.sh

nhop=3
drop_rate=1.0
. scripts/quick-start.sh
