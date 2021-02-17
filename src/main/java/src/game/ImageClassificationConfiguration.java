package src.game;


import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Random;

public class ImageClassificationConfiguration
{
    protected static long seed = 42;
    protected static Random rng = new Random(seed);
    protected static int listenerFreq = 1;
    protected static int nCores = 8;
    protected static int channels = 1;
    protected static int numLabels = 10;
    protected static int width = 28,height = 28;
    protected static long trainTime, testTime, startTime, endTime, totalTime;


    public static MultiLayerNetwork build()
    {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .miniBatch(false)
                .updater(Nesterovs.builder().learningRate(0.02).momentum(0.9).build())
                .l1(0.001)
                .l1Bias(0.002)
                .l2(0.01)
                .l2Bias(0.02)
                .list()
                .layer(0,new DenseLayer.Builder()
                        .name(" Input layer")
                        .nIn(28*28)
                        .nOut(256)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1,new DenseLayer.Builder()
                        .name(" Hidden layer 1")
                        .nIn(256)
                        .nOut(128)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(2,new DenseLayer.Builder()
                        .name(" Hidden layer 2")
                        .nIn(128)
                        .nOut(64)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(3,new OutputLayer.Builder()
                        .name(" Output Layer ")
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(64)
                        .nOut(10)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(28,28,1))
                .build();

        return new MultiLayerNetwork(conf);

    }
    public static MultiLayerNetwork getModelV1()  //snake variation
    {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .miniBatch(true)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .activation(Activation.RELU)
                //.updater(new Adam.Builder()
                        //
                        // .learningRate(0.01).beta1(0.9).beta2(0.999).build())
                .updater(new Nesterovs.Builder().momentum(0.9)
                        //.learningRateSchedule(new StepSchedule(ScheduleType.EPOCH,0.01,0.001,50))
                        .learningRate(0.002)
                        .build())
                .l2(1e-3)
                .l1(1e-3)
                //.dropOut(new Dropout(1e-7))
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(600)
                        .nOut(256)
                        .l1(1e-3)
                        .build())
                .layer(1,new DenseLayer.Builder()
                        .nOut(256)
                        .l1(1e-5)
                        .build())
                .layer(2,new OutputLayer.Builder()
                        .nOut(4)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(1, Configuration.BOARD_ROWS* Configuration.BOARD_COLUMNS ,1))
                .build();


        return new MultiLayerNetwork(conf);
    }
    public static MultiLayerNetwork getModelV2()
    {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .miniBatch(true)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .activation(Activation.RELU)
                //.updater(new Adam.Builder()
                //
                // .learningRate(0.01).beta1(0.9).beta2(0.999).build())
                .updater(new Nesterovs.Builder().momentum(0.9)
                        //.learningRateSchedule(new StepSchedule(ScheduleType.EPOCH,0.01,0.001,50))
                        .learningRate(0.002)
                        .build())
                .l2(1e-3)
                .l1(1e-3)
                //.dropOut(new Dropout(1e-7))
                .list()
                .layer(0,new ConvolutionLayer.Builder()
                        .nOut(256)
                        .l1(1e-3)
                        .padding(2,2)
                        .build())
                .layer(1,new DenseLayer.Builder()
                        .nOut(256)
                        .l1(1e-5)
                        .build())
                .layer(2,new OutputLayer.Builder()
                        .nOut(4)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(1,600,1))
                .build();


        return new MultiLayerNetwork(conf);
    }

    public static MultiLayerNetwork getModelV3()
    {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .miniBatch(true)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .activation(Activation.RELU)
                //.updater(new Adam.Builder()
                //
                // .learningRate(0.01).beta1(0.9).beta2(0.999).build())
                .updater(new Nesterovs.Builder().momentum(0.9)
                        //.learningRateSchedule(new StepSchedule(ScheduleType.EPOCH,0.01,0.001,50))
                        .learningRate(0.002)
                        .build())
                .l2(1e-3)
                .l1(1e-3)
                //.dropOut(new Dropout(1e-7))
                .list()
                .layer(0,new ConvolutionLayer.Builder()
                        .dataFormat(CNN2DFormat.NCHW)
                        .nOut(256)
                        .l1(1e-3)
                        .build())
                .layer(1, new SubsamplingLayer
                        .Builder(SubsamplingLayer.PoolingType.MAX)
                        .name("sub-sampling")
                        .stride(1,1)
                        .build())
                .layer(2,new DenseLayer.Builder()
                        .nOut(256)
                        .l1(1e-5)
                        .build())
                .layer(3,new OutputLayer.Builder()
                        .nOut(10)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1))
                .build();


        return new MultiLayerNetwork(conf);
    }

    public static MultiLayerNetwork getModelV4()
    {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam.Builder()
                        .learningRate(1e-3)
                        .build() )
                .list()
                .layer(0,new Convolution2D.Builder()
                        .nOut(32)
                        .name("conv2D_0")
                        .dataFormat(CNN2DFormat.NCHW)
                        .gradientNormalization(new BatchNormalization().getGradientNormalization())
                        .kernelSize(3,3)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new Convolution2D.Builder()
                        .nOut(32)
                        .name("conv2D_1")
                        .kernelSize(5,5)
                        .activation(Activation.RELU)
                        .gradientNormalization(new BatchNormalization().getGradientNormalization())
                        .build())
                .layer(2,new Convolution2D.Builder()
                        .nOut(32)
                        .name("conv2D_2")       //PADDING LAYER 2 E 5
                        .kernelSize(5,5)
                        .stride(2,2)
                        .padding(2,2)
                        .activation(Activation.RELU)
                        .gradientNormalization(new BatchNormalization().getGradientNormalization())
                        .dropOut(0.4)
                        .build())
                .layer(3,new Convolution2D.Builder()
                        .nOut(64)
                        .kernelSize(3,3)
                        .name("conv2D_3")
                        .gradientNormalization(new BatchNormalization().getGradientNormalization())
                        .activation(Activation.RELU)
                        .build())
                .layer(4,new Convolution2D.Builder()
                        .nOut(64)
                        .kernelSize(3,3)
                        .name("conv2D_4")
                        .gradientNormalization(new BatchNormalization().getGradientNormalization())
                        .activation(Activation.RELU)
                        .build())
                .layer(5,new Convolution2D.Builder()
                        .nOut(32)
                        .name("conv2D_5")
                        .kernelSize(5,5)
                        .stride(2,2)
                        .padding(2,2)
                        .activation(Activation.RELU)
                        .gradientNormalization(new BatchNormalization().getGradientNormalization())
                        .dropOut(0.4)
                        .build())
                .layer(6, new ConvolutionLayer.Builder()
                        .nOut(128)
                        .name("conv2D_6")
                        .kernelSize(4,4)
                        .activation(Activation.RELU)
                        .dropOut(0.4)
                        .gradientNormalization(new BatchNormalization().getGradientNormalization())
                        .build())
                .layer(7, new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1))
                .build();


        return new MultiLayerNetwork(conf);
    }

    public static MultiLayerNetwork mnistConfig()
    {
        int nChannels = 1;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .l1(0.01).l2(0.0005)
                .l1Bias(0.02).l2Bias(0.001)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs.Builder().learningRate(0.01).momentum(0.9).build())
                .list()
                .layer(0, new ConvolutionLayer
                        .Builder(5,5)
                        .name("convolution")
                        .nIn(nChannels)
                        .stride(1,1)
                        .nOut(20)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(1, new SubsamplingLayer
                        .Builder(SubsamplingLayer.PoolingType.MAX)
                        .name("sub-sampling")
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer
                        .Builder(5,5)
                        .name("convolution ")
                        .stride(1,1)
                        .nOut(50)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(3, new SubsamplingLayer
                        .Builder(SubsamplingLayer.PoolingType.MAX)
                        .name("sub-sampling")
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer
                        .Builder()
                        .name(" dense ")
                        .activation(Activation.LEAKYRELU)
                        .nOut(500)
                        .build())
                .layer(5, new OutputLayer
                        .Builder()
                        .name("output ")
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1))
                .backpropType(BackpropType.Standard)
                .build();

        return new MultiLayerNetwork(conf);
    }


    public static MultiLayerNetwork configuration()
    {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(784).nOut(200)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(200).nOut(10).build())
                .build();
        return new MultiLayerNetwork(conf);
    }

    private static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private static ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }

    private static ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
    }

    private static SubsamplingLayer maxPool(String name,  int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }

    private static DenseLayer fullyConnected(String name, int out, double bias, double dropOut, Distribution dist) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(dist).build();
    }
    public static MultiLayerNetwork LeNet()
    {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.005) // tried 0.0001, 0.0005
                .activation(Activation.RELU)
                .l1(0.0001) // tried 0.00001, 0.00005, 0.000001
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater( new Adam.Builder().learningRate(1e-2).build())
                //.updater(new Nesterovs.Builder().learningRate(0.1).momentum(0.9).build())
                .list()
                .layer(0, convInit("cnn1", channels, 100 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                .layer(1, maxPool("maxpool1", new int[]{1,1}))
                .layer(2, conv5x5("cnn2", 200, new int[]{5, 5}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxool2", new int[]{1,1}))
                .layer(4, new DenseLayer.Builder().nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }

    public static MultiLayerNetwork AlexNet() {
        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(new NormalDistribution(0.0, 0.01))
                .activation(Activation.RELU)
                .updater(new Nesterovs.Builder().momentum(0.9).learningRate(0.01).build())
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l1(1e-2)
                .l1Bias(1e-2*2)
                .weightDecay(1e-2)
                .l2(5 * 1e-4)
                .miniBatch(false)
                .list()
                .layer(0, convInit("cnn1", channels, 96, new int[]{11, 11}, new int[]{4, 4}, new int[]{1, 1}, 0))
                .layer(1, new LocalResponseNormalization.Builder().name("lrn1").build())
                .layer(2, maxPool("maxpool1", new int[]{1,1}))
                .layer(3, conv5x5("cnn2", 256, new int[] {1,1}, new int[] {2,2}, nonZeroBias))
                .layer(4, new LocalResponseNormalization.Builder().name("lrn2").build())
                .layer(5, maxPool("maxpool2", new int[]{1,1}))
                .layer(6,conv3x3("cnn3", 384, 0))
                .layer(7,conv3x3("cnn4", 384, nonZeroBias))
                .layer(8,conv3x3("cnn5", 256, nonZeroBias))
                .layer(9, maxPool("maxpool3", new int[]{1,1}))
                .layer(10, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(11, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(12, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);

    }

    public static MultiLayerNetwork VGGNet() {

        double nonZeroBias = 1;
        double dropOut = 0.5;


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) //
                .weightInit((new NormalDistribution(0.0, 0.01)))
                .activation(Activation.RELU)
                .updater(Updater.NESTEROVS)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l1(1e-1)
                .weightDecay(1e-1)
                .list()
                .layer(0, convInit("cnn1", channels, 64, new int[]{1, 1}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(1, maxPool("maxpool1", new int[]{2,2}))
                .layer(2, convInit("cnn2", 64, 128, new int[]{1, 1}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(3, maxPool("maxpool2", new int[]{2,2}))
                .layer(4, convInit("cnn3", 128, 128, new int[]{1, 1}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(5, convInit("cnn4", 128, 256, new int[]{1, 1}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(6, maxPool("maxpool3", new int[]{2,2}))
                .layer(7, convInit("cnn5", 256, 256, new int[]{1, 1}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(8, convInit("cnn6", 256, 512, new int[]{1, 1}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(9, maxPool("maxpool4", new int[]{2,2}))
                .layer(10, convInit("cnn7", 512, 512, new int[]{1, 1}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(11, convInit("cnn8", 512, 4096, new int[]{1, 1}, new int[]{1, 1}, new int[]{1, 1}, 0))
                .layer(12, maxPool("maxpool5", new int[]{2,2}))
                .layer(13, fullyConnected("ffn1", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(14, fullyConnected("ffn2", 4096, nonZeroBias, dropOut, new GaussianDistribution(0, 0.005)))
                .layer(15, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numLabels)
                        .activation(Activation.SOFTMAX)
                        .build())
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        return new MultiLayerNetwork(conf);
    }
}
