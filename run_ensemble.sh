#!/bin/sh
#set -ue

LANG=en_ewt
TYPE=pre

T1_TRAIN=data/T1/$LANG-ud-train.gold.conllu
T1_DEV=data/T1/$LANG-ud-dev.gold.conllu
T1_TEST=data/T1/$LANG-ud-test.gold.conllu
T2_TRAIN=data/T2/$LANG-ud-train_DEEP.gold.conllu
T2_DEV=data/T2/$LANG-ud-dev_DEEP.gold.conllu
T2_TEST=data/T2/$LANG-ud-test_DEEP.gold.conllu
UD_EVALUATE=data/UD/$LANG-ud-test.conllu

DIR=example/$LANG
mkdir -p $DIR



# Prepare train and dev data
# T1
python IMSurReal/align.py data/UD/$LANG-ud-train.conllu data/T1/$LANG-ud-train.conllu $T1_TRAIN
python IMSurReal/align.py data/UD/$LANG-ud-dev.conllu data/T1/$LANG-ud-dev.conllu $T1_DEV
python IMSurReal/align.py data/UD/$LANG-ud-test.conllu data/T1/$LANG-ud-test.conllu $T1_TEST test_file
# T2
python IMSurReal/align.py data/UD/$LANG-ud-train.conllu data/T2/$LANG-ud-train_DEEP.conllu $T2_TRAIN
python IMSurReal/align.py data/UD/$LANG-ud-dev.conllu data/T2/$LANG-ud-dev_DEEP.conllu $T2_DEV
python IMSurReal/align.py data/UD/$LANG-ud-test.conllu data/T2/$LANG-ud-test_DEEP.conllu $T2_TEST test_file


# Train T1
#linearization with TSP decoder
for i in $(seq 1 10)
do
echo Training T1 index: $i
python IMSurReal/main.py train -m $DIR/$LANG.t1.mdl$i.tsp.mdl -t $T1_TRAIN --d $T1_DEV --task tsp --paragraph $TYPE 
done

# (optional) swap post-processing, for treebanks with many non-projective trees
python IMSurReal/main.py train -m $DIR/$LANG.t1.swap.mdl -t $T1_TRAIN --d $T1_DEV --task swap
# inflection
python IMSurReal/main.py train -m $DIR/$LANG.t1.inf.mdl -t $T1_TRAIN --d $T1_DEV --task inf
# (optional) contraction, for some treebanks with contracted tokens
python IMSurReal/main.py train -m $DIR/$LANG.t1.con.mdl -t $T1_TRAIN --d $T1_DEV --task con

# Train T2
# linearization with TSP decoder

for i in $(seq 1 10)
do
echo Training T2 index: $i
python IMSurReal/main.py train -m $DIR/$LANG.t2.mdl$i.tsp.mdl -t $T2_TRAIN --d $T2_DEV --task tsp --paragraph $TYPE 
done

# (optional) swap post-processing, for treebanks with many non-projective trees
python IMSurReal/main.py train -m $DIR/$LANG.t2.swap.mdl -t $T2_TRAIN --d $T2_DEV --task swap
# completion (function words generation)
python IMSurReal/main.py train -m $DIR/$LANG.t2.gen.mdl -t $T2_TRAIN --d $T2_DEV --task gen 
# inflection
python IMSurReal/main.py train -m $DIR/$LANG.t2.inf.mdl -t $T2_TRAIN --d $T2_DEV --task inf
# (optional) contraction, for some treebanks with contracted tokens
python IMSurReal/main.py train -m $DIR/$LANG.t2.con.mdl -t $T2_TRAIN --d $T2_DEV --task con


# Predict Generate Paragraph
# Predict on test data
# T1
T1_output=()
for i in $(seq 1 10)
do
echo Predict T1 index: $i
python IMSurReal/main.py pred  -m $DIR/$LANG.t1.mdl$i.tsp.mdl -i $T1_TEST -p $DIR/$LANG.t1.mdl$i.tsp.conllu --paragraph $TYPE
T1_output[$i]="$DIR/$LANG.t1.mdl$i.tsp.conllu"
done
echo "${T1_output[*]}"
python IMSurReal/ensemble.py $DIR/$LANG.t1.tsp.conllu ${T1_output[*]}


python IMSurReal/main.py pred  -m $DIR/$LANG.t1.swap.mdl -i $DIR/$LANG.t1.tsp.conllu -p $DIR/$LANG.t1.swap.conllu
python IMSurReal/main.py pred  -m $DIR/$LANG.t1.inf.mdl -i $DIR/$LANG.t1.swap.conllu -p $DIR/$LANG.t1.inf.conllu
python IMSurReal/main.py pred  -m $DIR/$LANG.t1.con.mdl -i $DIR/$LANG.t1.inf.conllu -p $DIR/$LANG.t1.con.conllu

# T2
T2_output=()
for i in $(seq 1 10)
do
echo Predict T2 index: $i
python IMSurReal/main.py pred  -m $DIR/$LANG.t2.mdl$i.tsp.mdl -i $T2_TEST -p $DIR/$LANG.t2.mdl$i.tsp.conllu --paragraph $TYPE
T2_output[$i]="$DIR/$LANG.t2.mdl$i.tsp.conllu"
done
echo "${T2_output[*]}"
python IMSurReal/ensemble.py $DIR/$LANG.t2.tsp.conllu ${T2_output[*]}


python IMSurReal/main.py pred  -m $DIR/$LANG.t2.swap.mdl -i $DIR/$LANG.t2.tsp.conllu -p $DIR/$LANG.t2.swap.conllu
python IMSurReal/main.py pred  -m $DIR/$LANG.t2.gen.mdl -i $DIR/$LANG.t2.swap.conllu -p $DIR/$LANG.t2.gen.conllu
python IMSurReal/main.py pred  -m $DIR/$LANG.t2.inf.mdl -i $DIR/$LANG.t2.gen.conllu -p $DIR/$LANG.t2.inf.conllu
python IMSurReal/main.py pred  -m $DIR/$LANG.t2.con.mdl -i $DIR/$LANG.t2.inf.conllu -p $DIR/$LANG.t2.con.conllu

## evaluate test prediction (BLEU score on tokenized test)
python IMSurReal/evaluate.py $UD_EVALUATE $DIR/$LANG.t1.con.conllu
python IMSurReal/evaluate.py $UD_EVALUATE $DIR/$LANG.t2.con.conllu

