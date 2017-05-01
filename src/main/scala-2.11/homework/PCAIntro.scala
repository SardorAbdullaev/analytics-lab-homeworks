package homework

import org.apache.spark.ml.feature.{PCA, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.functions.{variance,sum}


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
    .csv("C:\\Users\\mrjus\\IdeaProjects\\Titanic\\src\\main\\resources\\homework\\hprice1.csv")
    .na.drop()

  val colsNotPrice = df.columns.filter(_ != "price")
  val assembler = new VectorAssembler()
    .setInputCols(colsNotPrice)
    .setOutputCol("features")

  val featureDf = assembler.transform(df).select("price","features")

  /*
  1. Fit a linear model using all variables (y = price)
  */
  val lr = new LinearRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
    .setLabelCol("price")
    .setFeaturesCol("features")
    .setPredictionCol("preds")

  val lrModel = lr.fit(featureDf)
  println("Linear MSE = "+lrModel.summary.rootMeanSquaredError)

  //2. Fit a linear model using just the first k PCs that “explain” >95% of the variance


  val pca = new PCA()
    .setInputCol("features")
    .setOutputCol("pcaFeatures")
    .setK(8)
    .fit(featureDf)

  val pcaDF = pca.transform(featureDf)
  val pcaVar = pcaDF.select(variance("pcaFeatures"))
  val pcaVarSum = pcaDF.select(sum(variance("pcaFeatures")))
  //TODO do scaling
  //for (i <- Range(0,8))
  //3. Compare test error of the models
}