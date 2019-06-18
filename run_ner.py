from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import json
import os
import time

import tensorflow as tf

from bert import modeling
from bert import optimization
from bert import tokenization

MIN_FLOAT = -1e30

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("bert_config_file", None, "The config json file corresponding to the pre-trained BERT model.")
flags.DEFINE_string("vocab_file", None, "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("data_dir", None, "The input data dir. Should contain the .json files for the task.")
flags.DEFINE_string("task_name", None, "The name of the task to train.")
flags.DEFINE_string("output_dir", None, "The output directory where the model checkpoints will be written.")
flags.DEFINE_string("export_dir", None, "The export directory where the saved model will be written.")

flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text. True for uncased models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run evaluation.")
flags.DEFINE_bool("do_predict", False, "Whether to run prediction.")
flags.DEFINE_bool("do_export", False, "Whether to run exporting.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 3.0, "Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1, "Proportion of training to perform linear learning rate warmup for.")

flags.DEFINE_integer("save_checkpoints_steps", 1000, "How often to save the model checkpoint.")
flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_integer("num_tpu_cores", 8,"Only used if `use_tpu` is True. Total number of TPU cores to use.")
flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from metadata.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self,
                 guid,
                 text,
                 label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids     

class NerProcessor(object):
    """Processor for the NER data set."""
    def __init__(self,
                 data_dir,
                 task_name):
        self.data_dir = data_dir
        self.task_name = task_name
    
    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        data_path = os.path.join(self.data_dir, "train-{0}".format(self.task_name), "train-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        return example_list
    
    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        data_path = os.path.join(self.data_dir, "dev-{0}".format(self.task_name), "dev-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        return example_list
    
    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        data_path = os.path.join(self.data_dir, "test-{0}".format(self.task_name), "test-{0}.json".format(self.task_name))
        data_list = self._read_json(data_path)
        example_list = self._get_example(data_list)
        return example_list
    
    def get_labels(self):
        """Gets the list of labels for this data set."""
        data_path = os.path.join(self.data_dir, "resource", "label.vocab")
        label_list = self._read_text(data_path)
        return label_list
    
    def _read_text(self,
                   data_path):
        if os.path.exists(data_path):
            with open(data_path, "rb") as file:
                data_list = []
                for line in file:
                    data_list.append(line.decode("utf-8").strip())

                return data_list
        else:
            raise FileNotFoundError("data path not found")
    
    def _read_json(self,
                   data_path):
        if os.path.exists(data_path):
            with open(data_path, "r") as file:
                data_list = json.load(file)
                return data_list
        else:
            raise FileNotFoundError("data path not found")
    
    def _get_example(self,
                     data_list):
        example_list = []
        for data in data_list:
            guid = data["id"]
            text = tokenization.convert_to_unicode(data["text"])
            label = tokenization.convert_to_unicode(data["label"])
            example = InputExample(guid=guid, text=text, label=label)
            example_list.append(example)
        
        return example_list

def convert_single_example(ex_index,
                           example,
                           label_list,
                           max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=[0] * max_seq_length)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    
    text_tokens = example.text.split(" ")
    label_tokens = example.label.split(" ")
    
    tokens = []
    labels = []
    for text_token, label_token in zip(text_tokens, label_tokens):
        text_sub_tokens = tokenizer.tokenize(text_token)
        label_sub_tokens = [label_token] + ["X"] * (len(text_sub_tokens) - 1)
        tokens.extend(text_sub_tokens)
        labels.extend(label_sub_tokens)
    
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[0:(max_seq_length - 2)]
    
    if len(labels) > max_seq_length - 2:
        labels = labels[0:(max_seq_length - 2)]
    
    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    
    input_tokens = []
    segment_ids = []
    label_ids = []
    
    input_tokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    
    for i, token in enumerate(tokens):
        input_tokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    
    input_tokens.append("[SEP]")
    segment_ids.append(0)
    label_ids.append(label_map["[SEP]"])
    
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    
    # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    input_mask = [1] * len(input_ids)
    
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids)
    return feature

def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)
        features.append(feature)
    
    return features

def input_fn_builder(features,
                     seq_length,
                     is_training,
                     drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []
    
    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_ids)
    
    def input_fn(params):
        batch_size = params["batch_size"]
        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids": tf.constant(all_input_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            "input_mask": tf.constant(all_input_mask, shape=[num_examples, seq_length], dtype=tf.int32),
            "segment_ids": tf.constant(all_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32),
            "label_ids": tf.constant(all_label_ids, shape=[num_examples, seq_length], dtype=tf.int32),
        })
        
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        
        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d
    
    return input_fn

def file_based_convert_examples_to_features(examples,
                                            label_list,
                                            max_seq_length,
                                            tokenizer,
                                            output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    def create_int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    
    writer = tf.python_io.TFRecordWriter(output_file)
    
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
    
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)
        
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_ids])
        
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        
        writer.write(tf_example.SerializeToString())
    
    writer.close()

def file_based_input_fn_builder(input_file,
                                seq_length,
                                is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }
    
    def _decode_record(record,
                       name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32. So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        
        return example
    
    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        
        d = d.apply(tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))
        
        return d
    
    return input_fn

def create_model(bert_config,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 label_list,
                 mode,
                 use_tpu):
    """Creates a NER model."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_tpu)
    
    # If you want to use sentence-level output, use model.get_pooled_output()
    # If you want to use token-level output, use model.get_sequence_output()
    result = model.get_sequence_output()
    result_mask = tf.cast(tf.expand_dims(input_mask, axis=-1), dtype=tf.float32)
    
    kernel_initializer = tf.glorot_uniform_initializer(seed=203, dtype=tf.float32)
    bias_initializer = tf.zeros_initializer
    dense_layer = tf.keras.layers.Dense(units=len(label_list), activation=None, use_bias=True,
        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
        kernel_regularizer=None, bias_regularizer=None, trainable=True)
    
    dropout_layer = tf.keras.layers.Dropout(rate=0.1, seed=981)
    
    result = dense_layer(result)
    if mode == tf.estimator.ModeKeys.TRAIN:
        result = dropout_layer(result)
    
    masked_result = result * result_mask + MIN_FLOAT * (1 - result_mask)
    predicts = tf.cast(tf.argmax(tf.nn.softmax(masked_result, axis=-1), axis=-1), dtype=tf.int32)
    
    loss = None
    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL] and label_ids is not None:
        label = tf.cast(label_ids, dtype=tf.float32)
        label_mask = tf.cast(input_mask, dtype=tf.float32)
        
        masked_label = tf.cast(label * label_mask, dtype=tf.int32)
        
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=masked_label, logits=masked_result)
        loss = tf.reduce_sum(cross_entropy * label_mask) / tf.reduce_sum(tf.reduce_max(label_mask, axis=-1))
    
    return loss, predicts

def model_fn_builder(bert_config,
                     label_list,
                     init_checkpoint,
                     learning_rate,
                     num_train_steps,
                     num_warmup_steps,
                     use_tpu):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features,
                 labels,
                 mode,
                 params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"] if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL] else None
        
        loss, predicts = create_model(bert_config, input_ids, input_mask, segment_ids, label_ids, label_list, mode, use_tpu)
        
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        
        if init_checkpoint:
            assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        
        if use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()
            
            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
        
        output_spec = None        
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(labels,
                          predicts,
                          label_list):
                label_map = {}
                for (i, label) in enumerate(label_list):
                    label_map[label] = i
                
                pad_id = tf.constant(label_map["[PAD]"], shape=[], dtype=tf.int32)
                out_id = tf.constant(label_map["O"], shape=[], dtype=tf.int32)
                x_id = tf.constant(label_map["X"], shape=[], dtype=tf.int32)
                cls_id = tf.constant(label_map["[CLS]"], shape=[], dtype=tf.int32)
                sep_id = tf.constant(label_map["[SEP]"], shape=[], dtype=tf.int32)
                
                masked_labels = (tf.cast(tf.not_equal(labels, pad_id), dtype=tf.int32) *
                    tf.cast(tf.not_equal(labels, out_id), dtype=tf.int32) *
                    tf.cast(tf.not_equal(labels, x_id), dtype=tf.int32) *
                    tf.cast(tf.not_equal(labels, cls_id), dtype=tf.int32) *
                    tf.cast(tf.not_equal(labels, sep_id), dtype=tf.int32))
                
                masked_predicts = (tf.cast(tf.not_equal(predicts, pad_id), dtype=tf.int32) *
                    tf.cast(tf.not_equal(predicts, out_id), dtype=tf.int32) *
                    tf.cast(tf.not_equal(predicts, x_id), dtype=tf.int32) *
                    tf.cast(tf.not_equal(predicts, cls_id), dtype=tf.int32) *
                    tf.cast(tf.not_equal(predicts, sep_id), dtype=tf.int32))
                
                precision = tf.metrics.precision(labels=masked_labels, predictions=masked_predicts)
                recall = tf.metrics.recall(labels=masked_labels, predictions=masked_predicts)
                
                metric = {
                    "precision": precision,
                    "recall": recall
                }
                
                return metric
            
            eval_metrics = (metric_fn, [label_ids, predicts, label_list])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={ "predicts": predicts },
                scaffold_fn=scaffold_fn)
        
        return output_spec
    
    return model_fn

def serving_input_fn():
    with tf.variable_scope("export"):
        features = {
            'input_ids': tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids'),
            'input_mask': tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask'),
            'segment_ids': tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')
        }
        
        return tf.estimator.export.build_raw_serving_input_receiver_fn(features)()

def write_to_file(data_list,
                  data_path):
    data_folder = os.path.dirname(data_path)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    
    with open(data_path, "w") as file:
        for data in data_list:
            file.write("{0}\n".format(data))

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError("Cannot use sequence length %d because the BERT model was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    
    tf.gfile.MakeDirs(FLAGS.output_dir)
    
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    
    data_dir = FLAGS.data_dir
    task_name = FLAGS.task_name.lower()
    processor = NerProcessor(data_dir, task_name)
    label_list = processor.get_labels()
    
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples()
        num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    
    model_fn = model_fn_builder(
        bert_config=bert_config,
        label_list=label_list,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu)
    
    # If TPU is not available, this will fall back to normal Estimator on CPU or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        export_to_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    
    if FLAGS.do_train:
        tf.logging.info("***** Run training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        
        train_features = convert_examples_to_features(
            examples=train_examples,
            label_list=label_list,
            max_seq_length=FLAGS.max_seq_length,
            tokenizer=tokenizer)

        train_input_fn = input_fn_builder(
            features=train_features,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples()
        tf.logging.info("***** Run evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            label_list=label_list,
            max_seq_length=FLAGS.max_seq_length,
            tokenizer=tokenizer)

        eval_input_fn = input_fn_builder(
            features=eval_features,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)
        
        result = estimator.evaluate(input_fn=eval_input_fn)
        print(result)
        precision = result["precision"]
        recall = result["recall"]
        f1_score = 2.0 * precision * recall / (precision + recall)
        
        tf.logging.info("***** Evaluation result *****")
        tf.logging.info("  Precision = %s", str(precision))
        tf.logging.info("  Recall = %s", str(recall))
        tf.logging.info("  F1 score = %s", str(f1_score))
    
    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples()
        tf.logging.info("***** Run prediction *****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        
        predict_features = convert_examples_to_features(
            examples=predict_examples,
            label_list=label_list,
            max_seq_length=FLAGS.max_seq_length,
            tokenizer=tokenizer)

        predict_input_fn = input_fn_builder(
            features=predict_features,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)
        
        result = estimator.predict(input_fn=predict_input_fn)
        predicts = [data["predicts"].tolist() for _, data in enumerate(result)]
        predicts = [([label_list[idx] for idx in predict], (predict + [0]).index(0)) for predict in predicts]
        predicts = [" ".join([token for token in predict[1:max_len-1] if token != "X"]) for predict, max_len in predicts]
        output_path = os.path.join(FLAGS.output_dir, "predict.{0}".format(time.time()))
        write_to_file(predicts, output_path)
    
    if FLAGS.do_export:
        tf.logging.info("***** Running exporting *****")
        tf.gfile.MakeDirs(FLAGS.export_dir)
        estimator.export_savedmodel(FLAGS.export_dir, serving_input_fn, as_text=False)

if __name__ == "__main__":
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("export_dir")
    tf.app.run()
