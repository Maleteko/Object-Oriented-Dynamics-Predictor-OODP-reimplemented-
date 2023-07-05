from tensorflow.keras import layers, Model, Input, backend
import tensorflow as tf

def scaling(item, size):
    return backend.resize_images(item, size, size, 'channels_last')

def bgExtractor():
    inputs = Input(shape=(80, 80, 3))
    conv1 = layers.Conv2D(64, (3, 3), padding="same")(inputs)
    conv1 =  layers.Activation('relu')(conv1)
    conv2 = layers.Conv2D(64, (3, 3), strides=2, padding="same")(conv1)
    conv2 = layers.Activation('relu')(conv2)
    conv3 = layers.Conv2D(64, (3, 3), strides=2, padding="same")(conv2)
    conv3 = layers.Activation('relu')(conv3)
    flat = layers.Flatten()(conv3)
    dense1 = layers.Dense(128)(flat)
    dense1 = layers.Activation('relu')(dense1)
    dense2 = layers.Dense(20*20*64)(dense1)
    dense2 = layers.Activation('relu')(dense2)
    reshape = layers.Reshape((20,20,64))(dense2)
    convt1 = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same")(reshape)
    convt1 = layers.Activation('relu')(convt1)
    convt2 = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same")(convt1)
    convt2 = layers.Activation('relu')(convt2)
    convt3 = layers.Conv2DTranspose(3, (3, 3), padding="same")(convt2)
    outputs = layers.Activation('tanh')(convt3)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def objectDetector():
    inputs = Input(shape=(80, 80, 3))
    conv1 = layers.Conv2D(64, (3, 3), strides=2, padding="same")(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 =  layers.Activation('relu')(conv1)
    conv2 = layers.Conv2D(64, (3, 3), strides=2, padding="same")(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 =  layers.Activation('relu')(conv2)
    conv3 = layers.Conv2D(64, (3, 3), padding="same")(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 =  layers.Activation('relu')(conv3)

    multiF = layers.concatenate([inputs, scaling(conv1,2), scaling(conv2,4), scaling(conv3,4)], 3)

    outputs = layers.Conv2D(1, (3, 3),  padding="same")((multiF))
    outputs = layers.BatchNormalization()(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def dynamicsNet(horizon, actions_num):
    inputs = Input(shape=(horizon, horizon, 3))
    conv1 = layers.Conv2D(16, (3, 3), strides=2, padding="same")(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 =  layers.Activation('relu')(conv1)
    conv2 = layers.Conv2D(32, (3, 3), strides=2, padding="same")(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 =  layers.Activation('relu')(conv2)
    conv3 = layers.Conv2D(64, (3, 3), padding="same")(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 =  layers.Activation('relu')(conv3)
    conv4 = layers.Conv2D(128, (3, 3), padding="same")(conv3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 =  layers.Activation('relu')(conv4)
    flat = layers.Flatten()(conv4)
    dense1 = layers.Dense(64)(flat)
    dense1 = layers.Activation('relu')(dense1)
    outputs = layers.Dense(2 * actions_num)(dense1)
    model = Model(inputs=inputs, outputs=outputs)
    return model



# taken from OODP repo

def calCoordinates(dynamicmask,inheight,batch_size):
    # co_dynamic : [bacthsize,2]
    Pmap = tf.reduce_sum(dynamicmask, axis=[1, 2])
    Pmap = tf.clip_by_value(Pmap,1e-10,1e10)
    Xmap = tf.tile(tf.reshape(tf.range(inheight),[1,inheight,1]),[batch_size,1,inheight])
    Ymap = tf.tile(tf.reshape(tf.range(inheight),[1,1,inheight]),[batch_size,inheight,1])
    x_dynamic = tf.reduce_sum(dynamicmask*tf.cast(Xmap,tf.float32),axis=[1, 2])/Pmap
    y_dynamic = tf.reduce_sum(dynamicmask*tf.cast(Ymap,tf.float32),axis=[1, 2])/Pmap

    return tf.stack([x_dynamic,y_dynamic],1)

def cropping(cx_a,cy_a,M_dynamic,M_static,batch_size,horizon):
    batch_idx = tf.tile(tf.reshape(tf.range(0, batch_size), [batch_size, 1, 1]), [1, horizon, horizon])
    x_idx = tf.tile(tf.reshape(tf.range(0, horizon), [1,  horizon, 1]), [batch_size, 1, horizon])+tf.reshape(cx_a,[-1,1,1])
    y_idx = tf.tile(tf.reshape(tf.range(0, horizon), [1, 1, horizon]), [batch_size, horizon, 1])+tf.reshape(cy_a,[-1,1,1])
    
    # objectes interaction area Mo_ia:[batch_size, horizon, horizon, object_maxnum ]
    # dynamic interaction area Ma_ia:[batch_size, horizon, horizon,1]
    indices = tf.stack([batch_idx, x_idx, y_idx], 3)

    padMa = tf.pad(M_dynamic,tf.cast([[0,0],[horizon/2,horizon/2],[horizon/2,horizon/2]],'int32'),"CONSTANT")
    padMo = tf.pad(M_static,tf.cast([[0,0],[0,0],[horizon/2,horizon/2],[horizon/2,horizon/2]],'int32'),"CONSTANT")
    padMo_tr=tf.transpose(padMo, [0, 2, 3, 1])

    Mo_ia = tf.gather_nd(padMo_tr, indices)
    Ma_ia = tf.gather_nd(padMa, indices)

    return Mo_ia,tf.expand_dims(Ma_ia,3)

def tailorModule(co_dynamic,M_dynamic,M_static,inheight,batch_size,horizon):
        
    # Mos : [batch_size, horizon, horizon, object_maxnum ]
    # Ma : [batch_size, horizon, horizon, 1 ]

    x0 = tf.cast(tf.floor(co_dynamic[:,0]), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(co_dynamic[:,1]), 'int32')
    y1 = y0 + 1
            
    zero = tf.zeros([], dtype='int32')
    max_x=tf.cast(inheight - 1, 'int32')
    max_y=tf.cast(inheight - 1, 'int32')

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')
    w1 = tf.reshape(((x1_f-co_dynamic[:,0]) * (y1_f-co_dynamic[:,1])), [-1,1,1,1])
    w2 = tf.reshape(((x1_f-co_dynamic[:,0]) * (co_dynamic[:,1]-y0_f)), [-1,1,1,1])
    w3 = tf.reshape(((co_dynamic[:,0]-x0_f) * (y1_f-co_dynamic[:,1])), [-1,1,1,1])
    w4 = tf.reshape(((co_dynamic[:,0]-x0_f) * (co_dynamic[:,1]-y0_f)), [-1,1,1,1])

    Mo1,Ma1 = cropping(x0,y0,M_dynamic,M_static,batch_size,horizon)
    Mo2,Ma2 = cropping(x0,y1,M_dynamic,M_static,batch_size,horizon)
    Mo3,Ma3 = cropping(x1,y0,M_dynamic,M_static,batch_size,horizon)
    Mo4,Ma4 = cropping(x1,y1,M_dynamic,M_static,batch_size,horizon)
    Mos = tf.add_n([tf.stop_gradient(w1)*Mo1, tf.stop_gradient(w2)*Mo2, tf.stop_gradient(w3)*Mo3, tf.stop_gradient(w4)*Mo4])
    Ma = tf.add_n([w1*Ma1, w2*Ma2, w3*Ma3, w4*Ma4])
    Ma = tf.stop_gradient(Ma)
    return Mos,Ma

def croppingForPred(ndx,ndy,I,M_dynamic,batch_size,inheight):
        
    # ndx, ndy = -dx, -dy : [batch_size]
    # I : [batch_size, channel, height, width]
    # M_dynamic : [batch_size, height, width]

    padI = tf.pad(I,[[0,0],[inheight,inheight],[inheight,inheight],[0,0]],"CONSTANT",-1.0)
    padMa= tf.pad(M_dynamic,[[0,0],[inheight,inheight],[inheight,inheight]],"CONSTANT",0.0)

    batch_idx = tf.tile(tf.reshape(tf.range(0, batch_size), [batch_size, 1, 1]), [1, inheight, inheight])
    # -dx+|dx|
    x_idx = tf.tile(tf.reshape(tf.range(0, inheight), [1,  inheight, 1]), [batch_size, 1, inheight])+tf.reshape(ndx+inheight,[-1,1,1])
    y_idx = tf.tile(tf.reshape(tf.range(0, inheight), [1, 1, inheight]), [batch_size, inheight, 1])+tf.reshape(ndy+inheight,[-1,1,1])
    
    # Pred_Ia : [batch_size, horizon, horizon, 3 ]
    # Pred_dynamicmask : [batch_size, horizon, horizon,1]
    indices = tf.stack([batch_idx, x_idx, y_idx], 3)

    Pred_Ia = tf.gather_nd(padI, indices)
    Pred_dynamicmask = tf.gather_nd(padMa, indices)
    
    return tf.cast(Pred_Ia,"float32"), tf.expand_dims(Pred_dynamicmask,3)

def STNmodule(dco,I,M_dynamic,inheight,batch_size):
    # dco dx,dy: [batch_size,2]
    # x : max_x-dx
    # y : max_y-dy

    x=inheight - 1-dco[:,0]
    y=inheight - 1-dco[:,1]

    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    zero = tf.zeros([], dtype='int32')
    max_x=tf.cast(inheight - 1, 'int32')
    max_y=tf.cast(inheight - 1, 'int32')

    x0 = tf.clip_by_value(x0, zero, 2*max_x)
    x1 = tf.clip_by_value(x1, zero, 2*max_x)
    y0 = tf.clip_by_value(y0, zero, 2*max_y)
    y1 = tf.clip_by_value(y1, zero, 2*max_y)

    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')
    w1 = tf.reshape(((x1_f-x) * (y1_f-y)), [-1,1,1,1])
    w2 = tf.reshape(((x1_f-x) * (y-y0_f)), [-1,1,1,1])
    w3 = tf.reshape(((x-x0_f) * (y1_f-y)), [-1,1,1,1])
    w4 = tf.reshape(((x-x0_f) * (y-y0_f)), [-1,1,1,1])

    Pred_Ia1,Pred_dynamicmask1 = croppingForPred(x0-max_x,y0-max_y,I,M_dynamic,batch_size,inheight)
    Pred_Ia2,Pred_dynamicmask2 = croppingForPred(x0-max_x,y1-max_y,I,M_dynamic,batch_size,inheight)
    Pred_Ia3,Pred_dynamicmask3 = croppingForPred(x1-max_x,y0-max_y,I,M_dynamic,batch_size,inheight)
    Pred_Ia4,Pred_dynamicmask4 = croppingForPred(x1-max_x,y1-max_y,I,M_dynamic,batch_size,inheight)
    Pred_Ia = tf.add_n([w1*Pred_Ia1, w2*Pred_Ia2, w3*Pred_Ia3, w4*Pred_Ia4])
    Pred_dynamicmask= tf.add_n([w1*Pred_dynamicmask1, w2*Pred_dynamicmask2, w3*Pred_dynamicmask3, w4*Pred_dynamicmask4])

    return Pred_Ia,Pred_dynamicmask