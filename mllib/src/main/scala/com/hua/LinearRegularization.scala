package com.hua

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.{LabeledPoint, LassoWithSGD, LinearRegressionWithSGD, RidgeRegressionWithSGD}
import org.apache.spark.mllib.linalg.Vectors

/**
  * Created by Administrator on 2017/1/7.
  */
object LinearRegularization {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RegressionRegularization").setMaster("local")
    val sc = new SparkContext(conf)
    //1.读取样本数据
    val data = sc.textFile("D:/mllib.txt") //文件中每一行就是RDD中的一个元素
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()

    //2.训练模型,新建线性回归模型，并设置训练参数
    val numIterations = 50
    val stepSize = 0.1
    val miniBatchFraction = 1.0
    //parsedData为输入样本，numIterations为迭代次数，stepSize为步长，miniBatchFraction为迭代因子
    val model = LinearRegressionWithSGD.train(parsedData, numIterations, stepSize)
    val model1 = RidgeRegressionWithSGD.train(parsedData, numIterations, stepSize, miniBatchFraction)
    val model2 = LassoWithSGD.train(parsedData, numIterations, stepSize, miniBatchFraction)
    println(model)
    //预测数据
    val v = Vectors.dense(Array(6.0,6.0,6.0))
    println("*****************预测的数据*****************")
    println("Prediction of Linear："+model.predict(v))
    println("Prediction of Ridge："+model1.predict(v))
    println("Prediction of Lasso："+model2.predict(v))
    println("*****************预测的数据*****************")
    //评价
    val valuesAndPreds = parsedData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE = valuesAndPreds.map { case (v, p) => math.pow((v - p), 2) }.mean() //mean()求平均值
    val valuesAndPredsRidge = parsedData.map { point =>
      val prediction = model1.predict(point.features)
      (point.label, prediction)
    }
    val MSE1 = valuesAndPredsRidge.map { case (v, p) => math.pow((v - p), 2) }.mean()
    val valuesAndPredsLasso = parsedData.map { point =>
      val prediction = model2.predict(point.features)
      (point.label, prediction)
    }
    val MSE2 = valuesAndPredsLasso.map { case (v, p) => math.pow((v - p), 2) }.mean()
    println("LinearRegressionModel training Mean Squared Error = " + MSE)
    println("RidgeRegressionModel training Mean Squared Error = " + MSE1)
    println("Lassomodel training Mean Squared Error = " + MSE2)
  }
}