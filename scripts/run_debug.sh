#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
bash generate_header/run.sh
bash $DIR/compile.sh debug
./build/examples/debug
