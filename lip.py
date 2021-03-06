import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, CSVLogger

from model import segunet
from generator import data_gen_small


def argparser():
    # command line argments
    parser = argparse.ArgumentParser(description="SegUNet LIP dataset")
    # parser.add_argument("--train_list",
    #         help="train list path")
    # parser.add_argument("--trainimg_dir",
    #         help="train image dir path")
    # parser.add_argument("--trainmsk_dir",
    #         help="train mask dir path")
    # parser.add_argument("--val_list",
    #         help="val list path")
    # parser.add_argument("--valimg_dir",
    #         help="val image dir path")impo
    # parser.add_argument("--valmsk_dir",
    #         help="val mask dir path")
    parser.add_argument("--batch_size",
            default=5,
            type=int,
            help="batch size")
    parser.add_argument("--n_epochs",
            default=10,
            type=int,
            help="number of epoch")
    parser.add_argument("--epoch_steps",
            default=6000,
            type=int,
            help="number of epoch step")
    parser.add_argument("--val_steps",
            default=1000,
            type=int,
            help="number of valdation step")
    parser.add_argument("--n_labels",
            default=20,
            type=int,
            help="Number of label")
    parser.add_argument("--input_shape",
            default=(512, 512, 3),
            help="Input images shape")
    parser.add_argument("--kernel",
            default=3,
            type=int,
            help="Kernel size")
    parser.add_argument("--pool_size",
            default=(2, 2),
            help="pooling and unpooling size")
    parser.add_argument("--output_mode",
            default="softmax",
            type=str,
            help="output activation")
    parser.add_argument("--loss",
            default="categorical_crossentropy",
            type=str,
            help="loss function")
    parser.add_argument("--optimizer",
            default="adadelta",
            type=str,
            help="oprimizer")
    parser.add_argument("--class_weights",
            default=True,
            help="dataset class weights")
    parser.add_argument("--gpu_num",
            default="0",
            type=str,
            help="num of gpu")
    args = parser.parse_args()

    return args

def main(args):
    # device number
    if args.gpu_num:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    print(os.environ["CUDA_VISIBLE_DEVICES"])
    # set the necessary directories
    train_dir = 'resources/train'
    test_dir = 'resources/test'

    train_list = []
    train_list_file = "resources/train_list.txt"
    val_list_file = "resources/val_list.txt"

    with open(train_list_file, "r") as f:
        for l in f: train_list.append(l.replace("\n", ""))
    val_list = []
    with open(val_list_file, "r") as f:
        for l in f: val_list.append(l.replace("\n", ""))


    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        session = tf.Session(config=config)
        KTF.set_session(session)
        KTF.set_learning_phase(1)

        # set callbacks
        cp_cb = ModelCheckpoint(
                filepath='resources/checkpoints/checkpoint',
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                mode='auto',
                period=2)
        es_cb = EarlyStopping(
                monitor='val_loss',
                patience=2,
                verbose=1,
                mode='auto')
        tb_cb = TensorBoard(
                log_dir='resources/logs/',
                write_images=True)
        csv_logger = CSVLogger('resources/logs/training.log')

        # set generater
        train_gen = data_gen_small('resources/train/',
                'resources/train/',
                train_list,
                args.batch_size,
                [args.input_shape[0], args.input_shape[1]],
                args.n_labels)
        val_gen = data_gen_small('resources/val/',
                'resources/val/',
                val_list,
                args.batch_size,
                [args.input_shape[0], args.input_shape[1]],
                args.n_labels)

        # set model
        model = segunet(args.input_shape, args.n_labels, args.kernel, args.pool_size, args.output_mode)
        print(model.summary())

        # compile model
        model.compile(loss=args.loss,
                optimizer=args.optimizer,
                metrics=["accuracy"])

        # fit with genarater
        model.fit_generator(generator=train_gen,
                steps_per_epoch=args.epoch_steps,
                epochs=args.n_epochs,
                validation_data=val_gen,
                validation_steps=args.val_steps,
                callbacks=[cp_cb, es_cb, tb_cb, csv_logger])

        model.save_weights("resources/weights/weights_01.hdf5")


if __name__ == "__main__":
    args =argparser()
    main(args)
