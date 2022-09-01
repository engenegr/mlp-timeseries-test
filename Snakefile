
rule test_model:
    input:
        "weights.hdf5",
        'test.txt',
        'test_labels.txt'
    output:
        report("confusion_matrix.png", category="ML Model")
    run:
        import tensorflow as tf
        import numpy as np
        import matplotlib.pyplot as plt
        from usermodule import get_base_model, plot_confusion_matrix
        from sklearn.metrics import confusion_matrix

        base_model = get_base_model()
        # Set cyclical learning rate
        optimizer = tf.keras.optimizers.Adam()
        base_model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

        base_model.load_weights('weights.hdf5')
        X_test = np.loadtxt("test.txt")
        y_test = np.loadtxt("test_labels.txt")
        base_model.evaluate(X_test,y_test)
        y_pred = (base_model.predict(X_test) > 0.5).astype("int32")

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test,y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix,classes=['SR', 'OTHER', 'AFIB'],
            title='Confusion matrix, without normalization')
        plt.savefig("confusion_matrix.png")


rule download_dataset:
    input:
        "ECG_Rhythm_Lead_I.csv"
    threads: 1
    run:
        import wget
        url = 'https://zenodo.org/record/5711347/files/ECG_Rhythm_Lead_I.csv'
        filename = wget.download(url, out="./")
        print(f"finished downloading {filename}")

rule prepare_dataset:
    input:
        "ECG_Rhythm_Lead_I.csv"
    output:
        'train.txt',
        'train_labels.txt',
        'val.txt',
        'val_labels.txt',
        'test.txt',
        'test_labels.txt',
        report("signal_example.png", category="Dataset"),
        report("label_hist.png", category="Dataset"),
    threads: 1
    run:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from usermodule.helpers import INPUT_SHAPE
        from sklearn.model_selection import train_test_split
        from sklearn import preprocessing

        dataset = pd.read_csv(str(input), skiprows=1, header=None)
        print(f"Dataset shape: {dataset.shape}, Labels {dataset.iloc[:, 6].unique()}")
        """ 
        head output
           Unnamed: 0  ecg_id   age  sex  height  weight rhythm_diag      0      1      2      3      4      5      6      7      8  ...    984    985    986    987    988    989    990    991    992    993    994    995    996    997    998    999
        0           0       1  56.0    1     NaN    63.0          SR -0.119 -0.116 -0.120 -0.117 -0.103 -0.097 -0.119 -0.096 -0.048  ...  0.131  0.113  0.154  0.189  0.228  0.219  0.198  0.194  0.115  0.107  0.107  0.106  0.090  0.069  0.086  0.022
        1           1       2  19.0    0     NaN    70.0       OTHER  0.004 -0.020 -0.053 -0.056 -0.062 -0.065 -0.061 -0.061 -0.064  ...  0.031 -0.002 -0.038 -0.017 -0.051 -0.054 -0.035 -0.045  0.004  0.044  0.507  0.554  0.316  0.121 -0.326 -0.348
        2           2       3  37.0    1     NaN    69.0          SR -0.029 -0.035 -0.054 -0.078 -0.088 -0.022  0.346  0.784  0.426  ...  0.070  0.046  0.025 -0.006 -0.040 -0.054 -0.040 -0.051 -0.026 -0.032 -0.052 -0.039 -0.034 -0.029 -0.048 -0.049
        3           3       4  24.0    0     NaN    82.0          SR -0.054 -0.053 -0.063 -0.060 -0.050 -0.054 -0.059 -0.058 -0.054  ... -0.014  0.018  0.057  0.275  0.486  0.217 -0.312 -0.511 -0.280 -0.076 -0.012  0.001 -0.003  0.026  0.026  0.028
        4           4       5  19.0    1     NaN    70.0          SR -0.034 -0.038 -0.057 -0.066 -0.080 -0.085 -0.058 -0.061 -0.068  ... -0.033 -0.027 -0.005  0.009  0.008  0.022  0.005  0.001  0.003  0.013  0.018 -0.001  0.007  0.000 -0.003 -0.012        
        """
        labels = {'SR': 0, 'OTHER': 1, 'AFIB': 2}
        reverse_labels = {v: k for k, v in labels.items()}
        ds_labels = dataset.iloc[:, 6].replace(reverse_labels)

        fig = plt.figure()
        plt.hist(ds_labels)
        plt.title("Distribution of the PTB Diagnostic ECG Database labels")
        plt.savefig("label_hist.png")
        #plt.close()
        print("The minimum and maximum values are {}, {}".format(np.min(dataset.iloc[:, 7:-2].values), np.max(dataset.iloc[:, 7:-2].values)))
        fig = plt.figure()

        #dataset = pd.DataFrame(, columns=dataset.columns, index=dataset.index)
        plt.plot(dataset.iloc[0, 7:(7+187)], label=f"{dataset.iloc[0, 6]}")
        plt.plot(dataset.iloc[0, (7+187):(7+2*187)], label=f"{dataset.iloc[0, 6]}")
        plt.plot(dataset.iloc[1, 7:(7 + 187)], label=f"{dataset.iloc[1, 6]}")
        plt.plot(dataset.iloc[2, 7:(7 + 187)], label=f"{dataset.iloc[2, 6]}")
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Non-normalized activation')
        plt.title("PTB Diagnostic ECG Database example signal")
        plt.savefig("signal_example.png")
        #plt.close()
        dataset.replace(labels, inplace=True)

        min_max_scaler = preprocessing.MinMaxScaler()
        input = dataset.iloc[:, 7:(7 + INPUT_SHAPE)]
        scaled = pd.DataFrame(min_max_scaler.fit_transform(input.T).T)
        labels = dataset.iloc[:, 6]

        X_train, X_val, y_train, y_val = train_test_split(
            scaled.values,
            labels.values,
            test_size=0.2,
            random_state=42)

        X_val, X_test, y_val, y_test = train_test_split(X_val,y_val, test_size=0.5,random_state=42)

        fig = plt.figure()
        plt.plot(X_train[0])
        plt.plot(X_train[1])
        plt.plot(X_train[2])
        plt.xlabel('Time')
        plt.ylabel('Normalized activation')
        plt.savefig("norm_signal_example.png")

        print("All features size {}".format(dataset.iloc[:, 7:].shape))
        print("Train features size {}".format(X_train.shape))
        print("Validation features size {}".format(X_val.shape))
        print("Test features size {}".format(X_test.shape))

        np.savetxt('train.txt', X_train)
        np.savetxt('train_labels.txt', y_train)
        np.savetxt('val.txt', X_val)
        np.savetxt('val_labels.txt', y_val)
        np.savetxt('test.txt', X_test)
        np.savetxt('test_labels.txt', y_test)


rule build_model:
    input:
        'train.txt',
        'train_labels.txt',
        'val.txt',
        'val_labels.txt'
    output:
        "weights.hdf5",
        report("model.png", category="ML Model"),
        report("LRFinder-learning.png", category="ML Model"),
        report("norm_signal_example.png", category="Dataset")
    run:
        import os

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        import numpy as np
        import tensorflow as tf
        import keras
        import matplotlib.pyplot as plt
        from usermodule import get_base_model, LRFinder, pretty_plot
        from tensorflow.keras.callbacks import ModelCheckpoint
        from tensorflow_addons.optimizers import CyclicalLearningRate

        tf.keras.utils.plot_model(get_base_model(),to_file="model.png")

        X_train = np.loadtxt(str(input[1]))
        y_train = np.loadtxt(str(input[2]))
        X_val = np.loadtxt(str(input[3]))
        y_val = np.loadtxt(str(input[4]))

        optimizer = keras.optimizers.Adam(lr=0.001)
        model = get_base_model()
        model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
        lr_finder = LRFinder(start_lr=1e-8,end_lr=1e-2,max_steps=100,smoothing=0.6)

        _ = model.fit(X_train,y_train,batch_size=128,epochs=150,callbacks=[lr_finder],verbose=False)

        fig = plt.figure()
        lr_finder.plot()
        plt.savefig("LRFinder-learning.png")

        # Set cyclical learning rate
        N = X_train.shape[0]
        batch_size = 256
        iterations = N / batch_size
        step_size = 2 * iterations

        lr_schedule = CyclicalLearningRate(1e-7,1e-5,step_size=step_size,scale_fn=lambda x: tf.pow(0.95,x))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        save_best_weights = ModelCheckpoint(filepath=str(output[0]),verbose=0,save_best_only=True)

        base_model = get_base_model()
        base_model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
        history = base_model.fit(X_train,y_train,
            validation_data=(X_val, y_val),
            shuffle=True,batch_size=batch_size,epochs=150,callbacks=[save_best_weights])

        #fig = plt.figure()
        #pretty_plot(history,'loss', lambda x: np.argmin(x),"history_loss.png")
        #pretty_plot(history,'accuracy', lambda x: np.argmax(x),"history_accuracy.png")
        #plt.show()



