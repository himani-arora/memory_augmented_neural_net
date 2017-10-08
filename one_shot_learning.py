from utils import OmniglotDataLoader, one_hot_decode, five_hot_decode
import tensorflow as tf
import argparse
import os
import numpy as np
from model import NTMOneShotLearningModel
from tensorflow.python import debug as tf_debug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--restore_training', default=False, action="store_true")
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--label_type', type=str, default="one_hot", help='one_hot or five_hot')
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--seq_length', type=int, default=50)
    parser.add_argument('--augment', default=False, action="store_true")
    parser.add_argument('--model', type=str, default="MANN", help='LSTM, MANN, MANN2 or NTM')
    parser.add_argument('--read_head_num', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epoches', type=int, default=100000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--rnn_size', type=int, default=200)
    parser.add_argument('--image_width', type=int, default=64)
    parser.add_argument('--image_height', type=int, default=64)
    parser.add_argument('--image_channel', type=int, default=3)
    parser.add_argument('--rnn_num_layers', type=int, default=1)
    parser.add_argument('--memory_size', type=int, default=128)
    parser.add_argument('--memory_vector_dim', type=int, default=40)
    parser.add_argument('--shift_range', type=int, default=1, help='Only for model=NTM')
    parser.add_argument('--write_head_num', type=int, default=1, help='Only for model=NTM. For MANN #(write_head) = #(read_head)')
    parser.add_argument('--test_batch_num', type=int, default=100)
    parser.add_argument('--n_train_classes', type=int, default=64)
    parser.add_argument('--n_test_classes', type=int, default=16)
    parser.add_argument('--test_frequency', type=int, default=100)
    parser.add_argument('--save_frequency', type=int, default=5000)
    parser.add_argument('--save_dir', type=str, default='./save/one_shot_learning')
    parser.add_argument('--tensorboard_dir', type=str, default='./summary/one_shot_learning')
    parser.add_argument('--embedder', type=str, default='', help='None/CNN')
    parser.add_argument('--embed_dim', type=int, default=128, help='Only when using CNN embedder')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)


def train(args):
    print("Initializing model...")
    model = NTMOneShotLearningModel(args)
    print("Initializing data loader...")
    data_loader = OmniglotDataLoader(
        image_size=(args.image_width, args.image_height, args.image_channel),
        n_train_classses=args.n_train_classes,
        n_test_classes=args.n_test_classes
    )
    b_start=0
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if args.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        if args.restore_training:
            saver = tf.train.Saver(max_to_keep=2)
            ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model)
            saver.restore(sess, ckpt.model_checkpoint_path)
            b_start=int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            print('Starting from step %d'%(b_start))
        else:
            saver = tf.train.Saver(tf.global_variables(),max_to_keep=2)
            tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(args.tensorboard_dir + '/' + args.model, sess.graph)
        print(args)
        print("batch\tloss\t1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th")
        for b in range(b_start, args.num_epoches):

            # Test

            if b % args.test_frequency == 0:
                x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,
                                                              type='test',
                                                              augment=args.augment,
                                                              label_type=args.label_type)
                feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y, model.is_training: False}
                output, learning_loss = sess.run([model.o, model.learning_loss], feed_dict=feed_dict)
                merged_summary = sess.run(model.learning_loss_summary, feed_dict=feed_dict)
                train_writer.add_summary(merged_summary, b)
                # state_list = sess.run(model.state_list, feed_dict=feed_dict)  # For debugging
                # with open('state_long.txt', 'w') as f:
                #     print(state_list, file=f)
                print('%d\t%.4f\t' % (b, learning_loss)),

                shot_eval=0
                accuracy = test_f(args, y, output)
                for accu in accuracy:
                    shot_eval+=1
                    shot_summary = tf.Summary(value=[tf.Summary.Value(tag='%d-shot'%(shot_eval),simple_value=accu)])
                    train_writer.add_summary(shot_summary, b)
                    print('%.4f\t' % accu),
                print('')

            # Save model

            if b % args.save_frequency == 0 and b > 0:
                saver.save(sess, args.save_dir + '/' + args.model + '/model.tfmodel', global_step=b)

            # Train

            x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,
                                                          type='train',
                                                          augment=args.augment,
                                                          label_type=args.label_type)
            feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y,model.is_training: True}
            sess.run(model.train_op, feed_dict=feed_dict)


def test(args):
    print("Initializing model...")
    model = NTMOneShotLearningModel(args)
    print("Initializing data loader...")
    data_loader = OmniglotDataLoader(
        image_size=(args.image_width, args.image_height),
        n_train_classses=args.n_train_classes,
        n_test_classes=args.n_test_classes
    )
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(args.save_dir + '/' + args.model)
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Test Result\n1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tloss")
        y_list = []
        output_list = []
        loss_list = []
        for b in range(args.test_batch_num):
            x_image, x_label, y = data_loader.fetch_batch(args.n_classes, args.batch_size, args.seq_length,
                                                          type='test',
                                                          augment=args.augment,
                                                          label_type=args.label_type)
            feed_dict = {model.x_image: x_image, model.x_label: x_label, model.y: y}
            output, learning_loss = sess.run([model.o, model.learning_loss], feed_dict=feed_dict)
            y_list.append(y)
            output_list.append(output)
            loss_list.append(learning_loss)
        accuracy = test_f(args, np.concatenate(y_list, axis=0), np.concatenate(output_list, axis=0))
        
        for accu in accuracy:
            print('%.4f\t' % accu),
        print(np.mean(loss_list))


#warning -> episode length must be >= 11
def test_f(args, y, output):
    correct = [0] * args.seq_length
    total = [0] * args.seq_length
    if args.label_type == 'one_hot':
        y_decode = one_hot_decode(y)
        output_decode = one_hot_decode(output)
    elif args.label_type == 'five_hot':
        y_decode = five_hot_decode(y)
        output_decode = five_hot_decode(output)
    for i in range(np.shape(y)[0]):
        y_i = y_decode[i]
        output_i = output_decode[i]
        # print(y_i)
        # print(output_i)
        class_count = {}
        for j in range(args.seq_length):
            if y_i[j] not in class_count:
                class_count[y_i[j]] = 0
            class_count[y_i[j]] += 1
            total[class_count[y_i[j]]] += 1
            if y_i[j] == output_i[j]:
                correct[class_count[y_i[j]]] += 1
    return [float(correct[i]) / total[i] if total[i] > 0. else 0. for i in range(1, 11)]


if __name__ == '__main__':
    main()
