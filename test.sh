BENCHMARK=$1
SHOT=$2
PI=$3
GPU=$4
FILENAME=$5

if [ ${BENCHMARK} == "pascal5i" ]
then
  DATA="pascal"
  SPLITS="0 1 2 3"
elif [ ${BENCHMARK} == "coco20i" ]
then
  DATA="coco"
  SPLITS="0 1 2 3"
elif [ ${BENCHMARK} == "pascal10i" ]
then
  DATA="pascal"
  SPLITS="10 11"
fi

printf "%s\nbenchmark: ${BENCHMARK}, shot: ${SHOT}, pi_estimation_strategy: ${PI}\n\n" "---" >> ${FILENAME}
for SPLIT in $SPLITS
do
  python3 -m src.test --config config/${DATA}.yaml \
            --opts split ${SPLIT} \
              shot ${SHOT} \
              pi_estimation_strategy ${PI} \
              n_runs 5 \
              gpus ${GPU} \
              |& tee -a ${FILENAME}
  printf "\n" >> ${FILENAME}
done
