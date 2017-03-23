package com.hua

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by Administrator on 2017/1/7.
  */
object Linear1 {
  //1 读取样本数据
  val conf=new SparkConf().setAppName("MroS1mmeSparkApp").setMaster("local")
  val sc=new SparkContext(conf)
  val data_path = "/user/tmp/lpsa.data"
  val data = sc.textFile(data_path)

  val examples = data.map { line =>
    val parts = line.split(',')
    LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
  }.cache()

  //2 样本数据划分训练样本与测试样本
  val splits = examples.randomSplit(Array(0.8, 0.2))
  val training = splits(0).cache()
  val test = splits(1).cache()
  val numTraining = training.count()
  val numTest = test.count()
  println(s"Training: $numTraining, test: $numTest.")

  //3 新建线性回归模型，并设置训练参数
  val numIterations = 100
  val stepSize = 1
  val miniBatchFraction = 1.0
  val model = LinearRegressionWithSGD.train(training, numIterations, stepSize, miniBatchFraction)

  //4 对测试样本进行测试
  val prediction = model.predict(test.map(_.features))
  val predictionAndLabel = prediction.zip(test.map(_.label))

  //5 计算测试误差
  val loss = predictionAndLabel.map {
    case (p, l) =>
      val err = p - l
      err * err
  }.reduce(_ + _)
  val rmse = math.sqrt(loss / numTest)
  println(s"Test RMSE = $rmse.")
}
