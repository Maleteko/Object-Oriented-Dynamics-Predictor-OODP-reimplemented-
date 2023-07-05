import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pathlib
from PIL import Image
import numpy as np
from glob import glob
import tensorflow as tf
import tqdm
import math
from tensorflow.keras import optimizers
from common import loss as LOSS

from common.model import bgExtractor, objectDetector, dynamicsNet, calCoordinates, tailorModule, STNmodule


data_frames_per_episode = 8 + 1 # +1 because of image, next image
data_episodes = 4096
levels = range(2)
validation_level = range(5,15)

test_data_size = 8
training_batchsize = 8
test_batchsize = test_data_size
validation_batchsize = len(validation_level)

epoch_num = 10
actions_num = 5

object_detector_num = (1,4) #(dynamic, static)
horizon=32

learning_rate = 0.0001



def get_images(dataset):
    images = []
    for data in dataset:
        filename = data.numpy() 
        image = Image.open(filename)
        images.append(np.array(image))
    images = np.array(images)/255
    return images

def output_to_image(output):
    output = output * 255
    images = []
    for img in output:
        images.append(Image.fromarray(img.astype(np.uint8)))
    return images

def load_actions(path):
    with open(path) as f:
        read_data = f.read()
        data = read_data.split("\n")
    return data[:-1]

def create_dataset():
    # dataset shape: [(img_path, next_img_path, action)]
    print("Create Dataset...")
    i = 0
    img_data = []
    next_img_data = []
    actions = []
    for i in tqdm.tqdm(range(data_episodes)):
        for level in levels:
            img_paths = ["data/frames/train/lvl"+ str(level)+ "_"+str(i)+"_{0:02}.png".format(j) for j in range(data_frames_per_episode)]
            img_data = img_data + img_paths[:-1]
            next_img_data = next_img_data + img_paths[1:]
            actions = actions + load_actions("data/frames/train/lvl"+ str(level)+ "_"+str(i)+"_actions")

    dataset = tf.data.Dataset.from_tensor_slices((img_data, next_img_data, actions))
    img_data = []
    next_img_data = []
    actions = []
    for level in validation_level:
            img_data = img_data + ["data/frames/validation/lvl"+ str(level)+ "_0_00.png"]
            next_img_data = next_img_data + ["data/frames/validation/lvl"+ str(level)+ "_0_01.png"]
            actions = actions + [load_actions("data/frames/validation/lvl"+ str(level)+ "_0_actions")[0]]
    validation_dataset = tf.data.Dataset.from_tensor_slices((img_data, next_img_data, actions))
    validation_dataset = validation_dataset.batch(validation_batchsize)
    test_dataset = dataset.take(test_data_size) 
    test_dataset = test_dataset.batch(test_data_size)
    train_dataset = dataset.skip(test_data_size)
    return validation_dataset, test_dataset, train_dataset

def train_on_batch(images, next_images, actions, batch_size, bg_model, object_model, dynamics_model, is_train=True):

    # background net
    bg = bg_model(images, training=is_train)
    next_bg = bg_model(next_images, training=is_train)
    
    #object detector
    objs = None
    for obj_model in object_model:
        if objs == None:
            objs = obj_model(images, training=is_train)
            next_objs = obj_model(next_images, training=is_train)
        else:
            objs = tf.concat([objs,obj_model(images, training=is_train)],3)
            next_objs = tf.concat([next_objs,obj_model(next_images, training=is_train)],3)
    objs = tf.transpose(objs,[0,3,1,2])
    objs = tf.nn.softmax(objs,axis= 1)
    next_objs = tf.transpose(next_objs,[0,3,1,2])
    next_objs = tf.nn.softmax(next_objs,axis= 1)

    # get dynamic object coordinates
    dyn_obj_coords = calCoordinates(objs[:,0], 80, batch_size)
    next_dyn_obj_coords = calCoordinates(next_objs[:,0], 80, batch_size)
    # get object masks
    Mos, _ = tailorModule(dyn_obj_coords, objs[:,0], objs[:,1:], inheight=80, batch_size=batch_size, horizon=horizon)
    # DynamicsNet
    MoNout = None
    for i, dyn_model in enumerate(dynamics_model):
        reMo=tf.reshape(Mos[:,:,:,i],[-1,horizon,horizon,1])
        x_idxes =tf.tile(tf.reshape(tf.range(-(horizon//2), horizon//2), [1,horizon, 1, 1]), [batch_size, 1, horizon,1])
        x_idxes = tf.cast(x_idxes,tf.float32)
        y_idxes = tf.tile(tf.reshape(tf.range(-(horizon//2), horizon//2), [1, 1,horizon,1]), [batch_size, horizon, 1,1])
        y_idxes = tf.cast(y_idxes,tf.float32)
        inputs = tf.concat([reMo,x_idxes,y_idxes],3)
        if i == 0:
            MoNout = tf.expand_dims(dyn_model(inputs, training=is_train), -1)
        else:
            MoNout = tf.concat([MoNout,tf.expand_dims(dyn_model(inputs, training=is_train),-1)],2)
    BkaNout = tf.Variable(tf.zeros([2*actions_num])) #  BkaNout : [batch_size,2] warum?
    tmp = tf.reduce_sum(MoNout, 2) #+ tf.expand_dims(BkaNout, 0) # tmp : [batch_size,2*action_dim]
    Pred_delt = tf.stack([tf.reduce_sum(tmp[:,:actions_num] * actions,1),tf.reduce_sum(tmp[:,actions_num:]*actions,1)],1) # Pred_delt : [batch_size,2]
    pred_co = Pred_delt + dyn_obj_coords

  #[batch_size, 3, horizon, horizon]
    M_dynamic = tf.transpose(tf.expand_dims(objs[:,0], 1),[0,2,3,1])
    M_dynamic_next = tf.expand_dims(next_objs[:, 0], 1)
    Recon_I = (1 - tf.expand_dims(objs[:, 0], -1)) * bg + tf.expand_dims(objs[:, 0], -1) * images
    Recon_nextI = (1 - tf.expand_dims(next_objs[:, 0], -1)) * next_bg + tf.expand_dims(next_objs[:, 0], -1) * next_images

    Pred_Ia, Pred_M_dynamic = STNmodule(Pred_delt, images, objs[:,0], 80, batch_size)

    dynamicproposal = tf.reduce_mean(tf.square(images - bg), axis=3) >= 0.2
    dynamicproposal = tf.cast(dynamicproposal, tf.float32)

    #groundtruth_dx =(self.truepos_next-self.truepos)*((120-1)/(120.0-1))
    #groundtruth_dx=tf.stack([groundtruth_dx[:,1],groundtruth_dx[:,0]],1)
    
  
    Dyn_pred = Pred_M_dynamic*Pred_Ia
    Pred_nextI = (1 - Pred_M_dynamic) * bg + Dyn_pred

    bg_loss = 1 * LOSS.bg_loss(bg, next_bg)
    pro_loss =  0 * LOSS.pro_loss(dynamicproposal, objs[:, 0])
    entropie_loss = 0.1 * LOSS.entropie_loss(objs)
    highway_loss = LOSS.highway_loss(next_dyn_obj_coords, pred_co, batch_size)
    prediction_loss = 100 * LOSS.prediction_loss(Pred_nextI, next_images)
    recon_loss = 100 * LOSS.recon_loss(Recon_I, images, Recon_nextI, next_images)
    consist_loss = 1 * LOSS.consist_loss(Pred_M_dynamic, M_dynamic_next)
    #GTMotion_loss = LOSS.GTMotion_loss(groundtruth_dx, Pred_delt, batch_size)

    return ([bg_loss, pro_loss, entropie_loss, highway_loss, prediction_loss, recon_loss, consist_loss],
            {"bg": bg, "objs": objs, "next_objs": next_objs, "Pred_nextI": Pred_nextI, "pred_coords": pred_co})

def test():
    #Test
    for image_path, next_image_path, actions in test_dataset:
        images = get_images(image_path)
        next_images = get_images(next_image_path)
        actions = tf.cast(tf.one_hot([int(x) for x in actions] ,actions_num),tf.float32)
        
        loss, test_images= train_on_batch(images, next_images, actions, test_batchsize, is_train=False)
        bg_loss, pro_loss, entropie_loss, highway_loss, prediction_loss, recon_loss, consist_loss = loss
        loss_sum = tf.math.reduce_sum(loss)

        print("---Test---\nLoss: " + str(loss_sum.numpy()) +
                        "\nbackground loss: " + str(bg_loss.numpy()) +
                        "\nentropie loss: "  + str(entropie_loss.numpy()) +
                        "\nprediction loss: " + str(prediction_loss.numpy()) +
                        "\nproporsal los: " + str(pro_loss.numpy()) +
                        "\nhighway loss: " + str(highway_loss.numpy()) +
                        "\nreconstruction loss: " + str(recon_loss.numpy()) +
                        "\nconsistency loss: " + str(consist_loss.numpy()))
        
        with tf.name_scope("test"):
            tf.summary.scalar("background_loss", bg_loss, step=step)
            tf.summary.scalar("entropie_loss", entropie_loss, step=step)
            tf.summary.scalar("prediction_loss",prediction_loss, step=step)
            tf.summary.scalar("proporsal_loss", pro_loss, step=step)
            tf.summary.scalar("highway_loss", highway_loss, step=step)
            tf.summary.scalar("reconstruction_loss", recon_loss, step=step)
            tf.summary.scalar("consistency_loss", consist_loss, step=step)
            #tf.summary.scalar("groundtruth_Motion loss", GTMotion_loss, step=step)
            tf.summary.scalar("all_loss", loss_sum, step=step)

        with tf.name_scope(".test"):
            with tf.name_scope(".Input"):
                tf.summary.image("Image",[images[0]],step=step)
                tf.summary.image("Next_Image",[next_images[0]],step=step)
            with tf.name_scope(".Test"):
                tf.summary.image("Predict_Next_Image",[test_images["Pred_nextI"][0]],step=step)
                tf.summary.image("Background_Image",[test_images["bg"][0]],step=step)
            with tf.name_scope("Objects"):
                tf.summary.image("Object_Image",[tf.transpose([x],[1,2,0])*images[0] for x in test_images["objs"][0]],step=step,max_outputs= 10)


def validation():
    #Validation
    for image_path, next_image_path, actions in validation_dataset:
        images = get_images(image_path)
        next_images = get_images(next_image_path)
        actions = tf.cast(tf.one_hot([int(x) for x in actions] ,actions_num),tf.float32)
        
        loss, test_images= train_on_batch(images, next_images, actions, validation_batchsize, is_train=False)
        bg_loss, pro_loss, entropie_loss, highway_loss, prediction_loss, recon_loss, consist_loss = loss
        loss_sum = tf.math.reduce_sum(loss)

        print("---Validation---\nLoss: " + str(loss_sum.numpy()) +
                        "\nbackground loss: " + str(bg_loss.numpy()) +
                        "\nentropie loss: "  + str(entropie_loss.numpy()) +
                        "\nprediction loss: " + str(prediction_loss.numpy()) +
                        "\nproporsal los: " + str(pro_loss.numpy()) +
                        "\nhighway loss: " + str(highway_loss.numpy()) +
                        "\nreconstruction loss: " + str(recon_loss.numpy()) +
                        "\nconsistency loss: " + str(consist_loss.numpy()))

        with tf.name_scope("validation"):
            tf.summary.scalar("background_loss", bg_loss, step=step)
            tf.summary.scalar("entropie_loss", entropie_loss, step=step)
            tf.summary.scalar("prediction_loss",prediction_loss, step=step)
            tf.summary.scalar("proporsal_loss", pro_loss, step=step)
            tf.summary.scalar("highway_loss", highway_loss, step=step)
            tf.summary.scalar("reconstruction_loss", recon_loss, step=step)
            tf.summary.scalar("consistency_loss", consist_loss, step=step)
            #tf.summary.scalar("groundtruth_Motion loss", GTMotion_loss, step=step)
            tf.summary.scalar("all_loss", loss_sum, step=step)
        for idx, level in enumerate(validation_level):
            with tf.name_scope("Level_"+str(level)):
                with tf.name_scope(".Input"):
                    tf.summary.image("Image",[images[idx]],step=step)
                    tf.summary.image("Next_Image",[next_images[idx]],step=step)
                with tf.name_scope(".Test"):
                    tf.summary.image("Predict_Next_Image",[test_images["Pred_nextI"][idx]],step=step)
                    tf.summary.image("Background_Image",[test_images["bg"][idx]],step=step)
                with tf.name_scope("Objects"):
                    tf.summary.image("Dynamic_Object_Image",[tf.transpose([test_images["objs"][idx][0]],[1,2,0])],step=step)
                    tf.summary.image("Object Image",[tf.transpose([test_images["objs"][idx][1]],[1,2,0])],step=step)
                    #tf.summary.image("Object Image",[tf.transpose([x],[1,2,0])*images[i] for i, x in enumerate(test_images["objs"][level])],step=step,max_outputs= 10)

def main():
    print("Create Models...")
    try: 
        bg_model = tf.keras.models.load_model('data/model/checkpoints/0/bg_model.h5', compile=False)
        print("background model restored")
    except:
        bg_model = bgExtractor()

    object_model = []
    try:
        for i in range(sum(object_detector_num)):
            object_model.append(tf.keras.models.load_model('data/model/checkpoints/0/object_model'+str(i)+'.h5', compile=False))
            print("object model restored")
    except:
        for i in range(sum(object_detector_num)):   
            object_model.append(objectDetector())

    dynamics_model = []
    try:
        for i in range(object_detector_num[1]):
            dynamics_model.append(tf.keras.models.load_model('data/model/checkpoints/0/dynamics_model'+str(i)+'.h5', compile=False))
        print("dynamic model restored")
    except:
        for i in range(object_detector_num[1]):
            dynamics_model.append(dynamicsNet(horizon, actions_num))
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    validation_dataset, test_dataset, train_dataset = create_dataset()
    train_data_size = len(list(train_dataset))

    print("Shuffle Trainingsdata...")
    train_dataset = train_dataset.shuffle(4096)

    print("Create Batches...")
    batch_num = int(math.ceil(train_data_size/training_batchsize))
    dataset = train_dataset.batch(training_batchsize)
    print("Start Training")
    step = 1
    writer = tf.summary.create_file_writer("data/train")
    with writer.as_default():
        for epoch in range(epoch_num):
            print("epoch " + str(epoch+1)+"/"+str(epoch_num))
            with tqdm.tqdm(total=batch_num) as bar:
                for image_path, next_image_path, actions in dataset:
                    images = get_images(image_path)
                    next_images = get_images(next_image_path)
                    actions = tf.cast(tf.one_hot([int(x) for x in actions] ,actions_num),tf.float32)
                    with tf.GradientTape(persistent=True) as tape:
                        
                        loss, _= train_on_batch(images, next_images, actions, training_batchsize)
                        bg_loss, pro_loss, entropie_loss, highway_loss, prediction_loss, recon_loss, consist_loss = loss
                        loss_sum = tf.math.reduce_sum(loss)

                    #update tensorboard and progress bar
                    with tf.name_scope(".training"):
                        tf.summary.scalar("background_loss", bg_loss, step=step)
                        tf.summary.scalar("entropie_loss", entropie_loss, step=step)
                        tf.summary.scalar("prediction_loss",prediction_loss, step=step)
                        tf.summary.scalar("proporsal_loss", pro_loss, step=step)
                        tf.summary.scalar("highway_loss", highway_loss, step=step)
                        tf.summary.scalar("reconstruction_loss", recon_loss, step=step)
                        tf.summary.scalar("consistency_loss", consist_loss, step=step)
                        #tf.summary.scalar("groundtruth_Motion loss", GTMotion_loss, step=step)
                        tf.summary.scalar("all_loss", loss_sum, step=step)
                    
                    if step % 1000 == 0:
                        test()

                    step += 1
                    bar.set_postfix(loss=loss_sum.numpy())
                    bar.update()
                    
                    #update models
                        #bg model
                    grads = tape.gradient(loss_sum, bg_model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, bg_model.trainable_weights))
                        #object model
                    for obj_model in object_model:
                        grads = tape.gradient(loss_sum, obj_model.trainable_weights)
                        optimizer.apply_gradients(zip(grads, obj_model.trainable_weights))
                        #dynamic model
                    for dyn_model in dynamics_model:
                        grads = tape.gradient(loss_sum, dyn_model.trainable_weights)
                        optimizer.apply_gradients(zip(grads, dyn_model.trainable_weights))
                    

            # save Checkpoints
            print("Save checkpoints")
            pathlib.Path("data/model/checkpoints/"+str(epoch)).mkdir(parents=True, exist_ok=True)
            bg_model.save("data/model/checkpoints/"+str(epoch)+"/bg_model.h5")
            for i, model in enumerate(object_model):
                model.save("data/model/checkpoints/"+str(epoch)+"/object_model"+str(i)+".h5")
            for i, model in enumerate(dynamics_model):
                model.save("data/model/checkpoints/"+str(epoch)+"/dynamics_model"+str(i)+".h5")

            validation()

    bg_model.save("data/model/bg_model.h5")
    for i, model in enumerate(object_model):
        model.save("data/model/object_model"+str(i)+".h5")
    for i, model in enumerate(dynamics_model):
        model.save("data/model/dynamics_model"+str(i)+".h5")
