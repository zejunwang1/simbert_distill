# simbert_distill: 一个两阶段 simbert 蒸馏方案

SimBERT (https://github.com/ZhuiyiTechnology/simbert) 是苏建林老师 2020 年开源的融检索与生成于一体的 BERT 模型。本项目提出了一个两阶段的 simbert 蒸馏方案，首先基于 MiniLMv2 进行通用场景下的预训练语言模型蒸馏，将 simbert base 蒸馏为一个层数较浅的小模型，然后在相似度任务上继续使用教师模型 simbert base 对该小模型进行蒸馏训练，得到最终的学生模型。

## Step1: 安装 easytokenizer

easytokenizer 是基于 c++ 编写的高效 BertTokenizer 实现，提供 python bindings，安装步骤如下：

```shell
git clone --recursive https://github.com/zejunwang1/simbert_distill
cd simbert_distill/easytokenizer/
python setup.py install
```

## Step2: MiniLMv2 通用蒸馏

MiniLMv2 原理介绍论文：[《MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers》](https://arxiv.org/abs/2012.15828)

MiniLMv2 是从层数深的 Transformer 类模型到层数较浅的 Transformer 类模型的蒸馏策略。它的优势是只需要取教师模型和学生模型中的各一层进行蒸馏训练，而不像其他方法需要蒸馏更多的层，从而避免更加复杂的 layer mapping 问题，并且效果优于 TinyBert 的蒸馏策略。

MiniLMv2 蒸馏的目标是教师模型某层的 query 与 query，key 与 key，value 与 value 的矩阵乘结果与学生模型最后一层的 query 与 query、key 与 key，value 与 value 的矩阵乘之间的 KL 散度损失。当教师模型为 large 版本时，选择倒数某一层，当教师模型是 base 版本时，选择最后一层进行蒸馏即可。

当教师模型为 large 版本时，head 数与学生模型不同，蒸馏目标的 shape 无法匹配，此时需要对 head 进行重组，先合并再按照统一的 num_relation_heads 重新分割 head_num 和 head_size。

使用 general_distill.py 进行通用场景下的蒸馏：

```shell
export CUDA_VISIBLE_DEVICES=0
simbert_base_dir=/home/wangzejun/nlp-tools/models/simbert-base-chinese/
python general_distill.py --training_file data/general_sents.txt \
                          --metrics_file data/general_metrics.json \
                          --vocab_file vocab.txt \
                          --teacher_model ${simbert_base_dir} \
                          --student_model student_config/ \
                          --output_dir student_init \
                          --from_scratch --epochs 3 \
                          --reduce_memory --do_lower_case \
                          --batch_size 256 \
                          --learning_rate 1e-4 \
                          --max_seq_length 128 \
                          --logging_steps 10 \
                          --save_steps 100
```

可支持的配置参数：

```
usage: general_distill.py [-h] --training_file TRAINING_FILE --metrics_file
                          METRICS_FILE --vocab_file VOCAB_FILE --teacher_model
                          TEACHER_MODEL --student_model STUDENT_MODEL
                          --output_dir OUTPUT_DIR
                          [--teacher_layer_index TEACHER_LAYER_INDEX]
                          [--student_layer_index STUDENT_LAYER_INDEX]
                          [--num_relation_heads NUM_RELATION_HEADS]
                          [--max_grad_norm MAX_GRAD_NORM]
                          [--local_rank LOCAL_RANK] [--seed SEED]
                          [--from_scratch] [--epochs EPOCHS] [--reduce_memory]
                          [--do_lower_case] [--batch_size BATCH_SIZE]
                          [--learning_rate LEARNING_RATE]
                          [--max_seq_length MAX_SEQ_LENGTH]
                          [--weight_decay WEIGHT_DECAY]
                          [--warmup_proportion WARMUP_PROPORTION]
                          [--logging_steps LOGGING_STEPS]
                          [--save_steps SAVE_STEPS]
```

其中参数含义如下：

- training_file：训练数据文件，每行为一个句子，形如 data/general_sents.txt 的形式。

- metrics_file：json 文件，存储 training_file 中的句子个数和最大序列长度信息，形如 data/general_metrics.json 的形式。

- vocab_file：教师模型和学生模型的词典文件。

- teacher_model：教师模型 simbert base 的文件夹路径。

- student_model：学生模型配置文件 config.json 所在的文件夹路径。

- output_dir：模型输出的目录。

- teacher_layer_index：可选，学生模型从教师模型学习的教师层，默认为 -1，表示教师模型的最后一层。

- student_layer_index：可选，学生模型从教师模型学习的学生层，默认为 -1，表示学生模型的最后一层。

- num_relation_heads：可选，重新组合之后的 head 数，默认为 12。

- max_grad_norm：可选，训练过程中梯度裁剪的 max_norm 参数，默认为 1.0。

- local_rank：可选，使用单机多卡分布式训练时的节点编号，默认为 -1。

- seed：可选，随机种子，默认为 42。

- from_scratch：可选，是否根据配置文件从零开始训练学生模型。

- epochs：可选，训练轮次，默认为 3。

- reduce_memory：可选，是否将训练数据存储在磁盘 memmaps 上，以大幅减少内存使用。

- do_lower_case：可选，是否进行英文字母大写转小写。

- batch_size：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数。默认为 64。

- learning_rate：可选，最大学习率，默认为 1e-4。

- max_seq_length：可选，输入到预训练模型中的最大序列长度，最大不能超过 512，默认为 128。

- weight_decay：可选，控制正则项力度的参数，用于防止过拟合，默认为 0.01。

- warmup_proportion：可选，学习率 warmup 策略的比例，如果为 0.1，则学习率会在前 10% 训练 steps 的过程中从 0 慢慢增长到 learning_rate，而后再缓慢衰减。默认为 0.01。

- logging_steps：可选，日志打印的间隔 steps，默认为 100。

- save_steps: 可选，保存模型参数的间隔 steps，默认为 500。

MiniLMv2 蒸馏得到的学生模型存放在 student_init 中。

## Step3: 句对相似度任务蒸馏

蒸馏的目标是教师模型计算得到的句对余弦相似度与学生模型得到的句对余弦相似度之间的均方误差损失。使用 task_distill_sentence_pairs.py 进行相似度任务蒸馏：

```shell
export CUDA_VISIBLE_DEVICES=0
simbert_base_dir=/home/wangzejun/nlp-tools/models/simbert-base-chinese/
python task_distill_sentence_pairs.py --training_file data/task_pairs.txt \
                                      --metrics_file data/task_metrics.json \
                                      --vocab_file vocab.txt \
                                      --teacher_model ${simbert_base_dir} \
                                      --student_model student_init/ \
                                      --output_dir checkpoint \
                                      --do_lower_case \
                                      --eval_file data/sts-b-dev.txt \
                                      --learning_rate 5e-5 \
                                      --train_batch_size 64 \
                                      --eval_batch_size 64 \
                                      --epochs 3 \
                                      --cosine_reduction 0.04 \
                                      --reduce_memory \
                                      --warmup_proportion 0.02 \
                                      --logging_steps 10 \
                                      --save_steps 100
```

可支持的配置参数：

```
usage: task_distill_sentence_pairs.py [-h] --training_file TRAINING_FILE
                                      --metrics_file METRICS_FILE --vocab_file
                                      VOCAB_FILE --teacher_model TEACHER_MODEL
                                      --student_model STUDENT_MODEL
                                      --output_dir OUTPUT_DIR
                                      [--eval_file EVAL_FILE]
                                      [--max_grad_norm MAX_GRAD_NORM]
                                      [--local_rank LOCAL_RANK] [--seed SEED]
                                      [--reduce_memory] [--from_scratch]
                                      [--epochs EPOCHS] [--do_lower_case]
                                      [--train_batch_size TRAIN_BATCH_SIZE]
                                      [--eval_batch_size EVAL_BATCH_SIZE]
                                      [--learning_rate LEARNING_RATE]
                                      [--max_seq_length MAX_SEQ_LENGTH]
                                      [--weight_decay WEIGHT_DECAY]
                                      [--warmup_proportion WARMUP_PROPORTION]
                                      [--logging_steps LOGGING_STEPS]
                                      [--save_steps SAVE_STEPS] [--save_best]
                                      [--cosine_reduction COSINE_REDUCTION]
```

大部分参数与 general_distill.py 中的含义相同，部分参数介绍如下：

- training_file：训练数据文件，每行为一个 '\t' 分隔的句子对，形如 data/task_pairs.txt 的形式。

- metrics_file：json 文件，存储 training_file 中的句子对个数和单个句子的最大序列长度信息，形如 data/task_metrics.json 的形式。

- eval_file：可选，评估数据文件，形如 data/sts-b-dev.txt 的形式，默认为 None。

- save_best：可选，是否保存最优评估数据集性能的 checkpoint。

- cosine_reduction：可选，教师模型句对余弦相似度减小量，默认为 0。当该参数大于 0 时，对教师模型计算得到的余弦相似度减去一个较小值，再与学生模型得到的结果进行损失计算。

## 轻量化模型

蒸馏得到的轻量化模型保存在 distilled_model 中，同时支持直接使用 HuggingFace transformers 进行加载：

| model_name                       | link                                                    |
| -------------------------------- | ------------------------------------------------------- |
| WangZeJun/simbert_distill_4L192H | https://huggingface.co/WangZeJun/simbert_distill_4L192H |

## Contact

邮箱： [wangzejunscut@126.com](mailto:wangzejunscut@126.com)

微信：autonlp
