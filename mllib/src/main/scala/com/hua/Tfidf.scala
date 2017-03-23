package com.hua

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
/**
  * Created by Administrator on 2017/1/6.
  */
object Tfidf {
  def main(args: Array[String]): Unit = {
    val conf=new SparkConf().setAppName("MroS1mmeSparkApp").setMaster("local")
    val sc=new SparkContext(conf)
    val sqlc = new SQLContext(sc)
    //创建实例数据
    val sentenceDataFrame = sqlc.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (1, "I wish Java could use case classes"),
      (2, "Logistic,regression,models,are,neat")
    )).toDF("label", "sentence")
    sentenceDataFrame.show()
    //句子转换成单词数组
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized.show()
    tokenized.select("words", "label","sentence").take(3).foreach(println)
    tokenized.select("words", "label").take(2).foreach(println)

    // val regexTokenizer = new RegexTokenizer()
    //   .setInputCol("sentence")
    //   .setOutputCol("words")
    //   .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)
    // val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    // regexTokenized.show()
    // regexTokenized.select("words", "label").take(3).foreach(println)

    // hashing计算TF值,同时还把停用词(stop words)过滤掉了. setNumFeatures(20)表最多20个词
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
    val featurizedData = hashingTF.transform(tokenized)
    featurizedData.show()

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.show()

    // 提取该数据中稀疏向量的数据,稀疏向量:SparseVector(size,indices,values)
    // rescaledData.select("features").rdd.map(row => row.getAs[linalg.Vector](0)).map(x => x.toSparse.indices).collect
    rescaledData.select("features", "label").take(3).foreach(println)
    //    [(20,[5,6,9],[0.0,0.6931471805599453,1.3862943611198906]),0]
    //    [(20,[3,5,12,14,18],[1.3862943611198906,0.0,0.6931471805599453,0.6931471805599453,0.6931471805599453]),1]
    //    [(20,[5],[0.0]),2]
    // 其中,20是标签总数,下一项是单词对应的hashing ID.最后是TF-IDF结果
  }
}
