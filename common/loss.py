import tensorflow as tf

def bg_loss(bg, next_bg):
    return tf.reduce_mean(tf.square(bg-next_bg))

#left out
def pro_loss(dynamicproposal, M_dynamic):
    weight = tf.math.reduce_sum(dynamicproposal, axis=[1, 2], keepdims=True) / (120 * 120)
    weightmask = dynamicproposal * (1 - weight) + (1 - dynamicproposal) * weight
    return tf.reduce_mean(tf.reduce_sum(tf.square(M_dynamic - dynamicproposal) * weightmask, axis = [1, 2]))
  
def entropie_loss(objs):
    return -tf.reduce_mean(tf.reduce_sum(objs * tf.math.log(tf.clip_by_value(objs, 1e-10, 1.0)),axis=1))

def highway_loss(co_dynamic_next, pred_co, batch_size):
    #return tf.math.log(1+(0.01 * tf.reduce_sum(tf.square(co_dynamic_next - pred_co)) / (2 * batch_size)))
    return tf.reduce_sum(tf.square(co_dynamic_next - pred_co)) / (2 * batch_size)

def prediction_loss(Pred_nextI, next_images):
    return tf.reduce_mean(tf.square(Pred_nextI - next_images))

def recon_loss(Recon_I, images, Recon_nextI, next_images):
    return 0.5 * tf.reduce_mean(tf.square(Recon_I - images)) + 0.5 * tf.reduce_mean(tf.square(Recon_nextI - next_images))

def consist_loss(Pred_M_dynamic, M_dynamic_next):
    return tf.reduce_mean(tf.square(tf.transpose(Pred_M_dynamic, [0,3,1,2]) - M_dynamic_next))

def GTMotion_loss(groundtruth_dx, Pred_delt, batch_size):
    return tf.reduce_sum(tf.square(groundtruth_dx - Pred_delt)) / (2 * batch_size)