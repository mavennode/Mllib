package com.hua

import java.util.logging.Logger

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression

/**
  * Created by Administrator on 2017/1/7.
  */
object Linear {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Regression").setMaster("local")
    val sc = new SparkContext(conf)
    //1.读取样本数据
    val data = sc.textFile("D:/mllib.txt") //文件中每一行就是RDD中的一个元素
    //  val data = sc.makeRDD(List(1.0,5.0,8.9,9.0))

    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()
    //样本数据划分训练样本和测试样本
    //由于randomSplit的第一个参数weights中传入的值有2个，因此，就会切分成2个RDD
    //把原来的rdd按照权重0.8,0.2，随机划分到这2个RDD中，权重高的RDD，划分到//的几率就大一些。
    //注意，权重的总和加起来为1，否则会不正常
    //  val splits = parsedData.randomSplit(Array(0.8,0.2))     //根据weights权重将一个RDD切分成多个RDD
    //  val training = splits(0).cache()
    //  val test = splits(1).cache()
    //  val numTraining = training.count()
    //  val numTest = test.count()
    //  println(s"Training: $numTraining, test: $numTest.")

    //3.训练模型,新建线性回归模型，并设置训练参数
    val numIterations = 100
    val stepSize = 0.0000001
    val miniBatchFraction = 1.0
    //parsedData为输入样本，numIterations为迭代次数，stepSize为步长，miniBatchFraction为迭代因子
    val model = LinearRegressionWithSGD.train(parsedData, numIterations, stepSize, miniBatchFraction)
    print("****************"+model)
    //4.对测试样本进行测试
    //  val prediction = model.predict(test.map(_.features))
    //  val predictionAndLabel = prediction.zip(test.map(_.label))
    //评价
    val valuesAndPreds = parsedData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE = valuesAndPreds.map { case (v, p) => math.pow((v - p), 2) }.mean() //mean()求平均值
    println("training Mean Squared Error = " + MSE)
    // 5.计算测量误差
    //  val loss = predictionAndLabel.map {
    //    case (p, l) =>
    //      val err = p - l
    //      err * err
    //  }.reduce(_ + _)
    //  val rmse = math.sqrt(loss / numTest)
    //  println(s"Test RMSE = $rmse.")
  }
}
