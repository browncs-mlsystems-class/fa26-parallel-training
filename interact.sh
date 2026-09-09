#!/bin/bash


PARTITION="gpus"
GRES="gpu:1"
CPUS="8"      # e.g. ~1 CPU per GPU, adjust based on your cluster's guidance
MEM="16G"     # scale up accordingly
TIME="10:00:00"
CONSTRAINT="--constraint=gtx_1080_ti|gtx_2080_ti|titan_rtx|rtx_a6000|l40"

while [[ $# -gt 0 ]]; do
  case $1 in
    --time)
      TIME="$2"
      shift 2
      ;;
    --mem)
      MEM="$2"
      shift 2
      ;;
	--gpus)
      if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: --gpus must be a positive integer, got '$2'"
        exit 1
      fi
      GRES="gpu:$2"
      shift 2
      ;;
    --cpus)
      CPUS="$2"
      shift 2
      ;;
    --constraint)
      CONSTRAINT="--constraint=$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Debug: TIME=$TIME MEM=$MEM CPUS=$CPUS GRES=$GRES CONSTRAINT=$CONSTRAINT"

# Launch an interactive job with specified options
srun \
  --partition=$PARTITION \
  --gres=$GRES \
  --time="$TIME" \
  --mem="$MEM" \
  --cpus-per-task="$CPUS" \
  $CONSTRAINT \
  --pty \
  $SHELL
