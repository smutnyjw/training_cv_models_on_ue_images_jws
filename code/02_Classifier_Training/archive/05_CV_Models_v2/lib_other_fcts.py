'''
File:   
Author: 
Date:   
Description: 

Other Notes:

'''
import keras


def change_layers_trainability(model: keras.Model,
                               layers_to_train,
                               layers_to_freeze):
    num_layers = len(model.layers)

    for x in range(num_layers):
        if x in layers_to_freeze:
            model.layers[x].trainable = False
        elif x in layers_to_train:
            model.layers[x].trainable = True
        else:
            print(f"*** WARN: Layer #{x} not specified in either to_train or "
                  f"to_freeze. Freezing layer")
            model.layers[x].trainable = False

    return model


# TODO - go ahead with plan for 'Run Customibility'? Change fct called based
#  on parameter?

def init_multiclass_vgg16(img_size):
    print("TODO")
    # Use loss=CategoricalCrossEntropy
    #
    # model.compile(optimizer="adam",
    #               loss="CategoricalCrossEntropy",
    #               metrics=[keras.metrics.CategoricalAccuracy,
    #                        keras.metrics.TopKCategoricalAccuracy(k=3),
    #                        keras.metrics.Precision])


def build_vgg16_og_1000():
    img_size = (224, 224, 3)
    model = keras.applications.VGG16(
        input_shape=img_size,
        include_top=True,
        weights="imagenet",
        pooling="avg",
        classes=1000,
        classifier_activation="softmax"
    )

    out.output_model_arch('./output/vgg16_og_architecture.txt', model)
    model.save_weights("./weights/vgg16_og_trained_weights.h5")

    return model, img_size
