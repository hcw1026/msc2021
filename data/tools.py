import tensorflow as tf
import collections
import enum
from functools import partial

ClassificationDescription = collections.namedtuple(
    "ClassificationDescription",
    ["tr_input", "tr_output", "val_input", "val_output"])

def unpack_data(problem_instance):
    if isinstance(problem_instance, ClassificationDescription):
        return list(problem_instance)
    return problem_instance

class StrEnum(enum.Enum):
    """an Enum represented by a string"""

    def __str__(self):
          return self.value

    def __repr__(self):
         return self.__str__()

class MetaSplit(StrEnum):
    """meta-datasets split supported by the DataProvider class"""
    TRAIN = "train"
    VALID = "val"
    TEST = "test"
    TRIAL = "trial" # for debugging

def normalise(tr_input, tr_output, val_input, val_output):
    """normalise tr.input and val.input independently"""
    tr_input = tf.nn.l2_normalize(tr_input, axis=-1)
    val_input = tf.nn.l2_normalize(val_input, axis=-1)
    return tr_input, tr_output, val_input, val_output

def description_map(tr_input, tr_output, val_input, val_output, description):
    """map list into description"""
    return description(tr_input, tr_output, val_input, val_output)

def shuffle_map(tr_input, tr_output, val_input, val_output):
    """shuffle within problem instance"""
    tr_size = tf.shape(tr_input)[0]
    val_size = tf.shape(val_input)[0]
    
    tr_indices = tf.random.shuffle(tf.range(0, tr_size, dtype=tf.int32))
    val_indices = tf.random.shuffle(tf.range(0, val_size, dtype=tf.int32))
    tf.print("1a",tf.shape(tr_input))
    tf.print("1b", tf.shape(tr_output))

    tr_input = tf.gather(tr_input, tr_indices)
    tr_output = tf.gather(tr_output, tr_indices)
    val_input = tf.gather(val_input, val_indices)
    val_output = tf.gather(val_output, val_indices)
    tf.print("2",tr_output)
    return tr_input, tr_output, val_input, val_output

def GenerateDataset(generator, output_signature, batch_size, description, shuffle):
    """data input pipeline for a generated tf.data.Dataset"""
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.repeat()
    if shuffle is True:
        dataset = dataset.map(shuffle_map, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False) # batch
    dataset = dataset.map(normalise, num_parallel_calls=tf.data.AUTOTUNE) #normalise
    dataset = dataset.map(partial(description_map, description=description), 
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
