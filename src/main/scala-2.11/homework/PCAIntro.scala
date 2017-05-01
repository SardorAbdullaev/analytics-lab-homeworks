package homework

import org.apache.spark.ml.feature.{PCA, StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{Column, Row}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.{sum, variance}


/**
  * Created by Sardor on 4/29/2017.
  */

//PCA intro

object PCAIntro {

  //TASK housing, regression
  import org.apache.spark.sql.{DataFrame, SparkSession}

  val sparkSession: SparkSession = SparkSession.builder.
    master("local")
    .appName("spark session example")
    .getOrCreate()
  import sparkSession.implicits._

  val df: DataFrame = sparkSession
    .read.option("header", "true").option("inferSchema", "true")
    .csv("C:\\Users\\mrjus\\IdeaProjects\\Analytics_lab_homework\\src\\main\\resources\\homework\\hprice1.csv")
    .na.drop()

  val colsNotPrice = df.columns.filter(_ != "price")
  val assembler = new VectorAssembler()
    .setInputCols(colsNotPrice)
    .setOutputCol("features")

  val featureDf = assembler.transform(df).select("price","features")

  val scaler = new StandardScaler()
    .setInputCol("features")
    .setOutputCol("features_scaled")

  val scalerModel = scaler.fit(featureDf)

  val scaledDf = scalerModel.transform(featureDf)
  /*
  1. Fit a linear model using all variables (y = price)
  */
  val lr = new LinearRegression()
    .setLabelCol("price")
    .setFeaturesCol("features_scaled")
    .setPredictionCol("preds")

  val lrModel = lr.fit(scaledDf)
  println("Linear MSE = "+lrModel.summary.rootMeanSquaredError)

  //2. Fit a linear model using just the first k PCs that “explain” >95% of the variance

  val pca = new PCA()
    .setInputCol("features_scaled")
    .setOutputCol("pcaFeatures")
    .setK(8)
    .fit(scaledDf)

  val pcaDF = pca.transform(scaledDf)

  def vectorHead(n:Int): UserDefinedFunction = udf{ x:DenseVector => x(n) }
  val readyDf =
    (0 until 8).foldLeft(pcaDF){
      (r,i)=>
        r.withColumn("PC"+(i+1).toString, vectorHead(i)(pcaDF("pcaFeatures")))
  }

  val total_var = (0 until 8).foldLeft(0.0)((r,i)=>
    r+readyDf.select(variance("PC"+(i+1).toString)).persist().head().getDouble(0)
  )
  (0 until 8).foreach {
    i =>
      val cumval = (0 until i).foldLeft(0.0)((r, i2) => r + readyDf.select(variance("PC" + (i2 + 1).toString)).head().getDouble(0))
      println(cumval / total_var)
      println("#########################")
  }
  val assembler2 = new VectorAssembler()
    .setInputCols((1 until 6).map(i=>"PC"+i).toArray)
    .setOutputCol("five_pca")

  val finalDF = assembler2.transform(readyDf)

  val PCAr = new LinearRegression()
    .setLabelCol("price")
    .setFeaturesCol("five_pca")
    .setPredictionCol("preds")

  val PCAlrModel = PCAr.fit(finalDF)
  println("PCE MSE = "+PCAlrModel.summary.rootMeanSquaredError)

  //3. Compare test error of the models
  println("Diff MSE = "+(PCAlrModel.summary.rootMeanSquaredError - lrModel.summary.rootMeanSquaredError))
  //Linear MSE = 12.164661372084828
  //PCA MSE = 31.07825959295135
  //Diff MSE = 18.913598220866525
}