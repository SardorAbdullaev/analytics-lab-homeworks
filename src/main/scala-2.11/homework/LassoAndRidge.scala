package homework

import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Random
/**
  * Created by Sardor on 4/26/2017.
  */
object LassoAndRidge {

  val sparkSession: SparkSession = SparkSession.builder.
    master("local")
    .appName("spark session example")
    .getOrCreate()
  import sparkSession.implicits._
//read file and remove NAs
  val df: DataFrame = sparkSession
    .read.option("header", "true").option("inferSchema", "true")
    .csv("C:\\Users\\mrjus\\IdeaProjects\\Analytics_lab_homework\\src\\main\\resources\\homework\\Hitters.csv")
    .na.drop()
    .filter($"Salary"!=="NA")

  val allColumns = df.columns.toList
  val categoryColumns = List("League","Division","NewLeague")
  val salaryColumn = "Salary"

  val assembler = new VectorAssembler()
    .setInputCols(allColumns.diff(categoryColumns).diff(Seq(salaryColumn)).toArray)
    .setOutputCol("features_for_scaling")

  val featuresForScalingDf = assembler.transform(df)

  val scaler = new StandardScaler()
    .setInputCol("features_for_scaling")
    .setOutputCol("features_scaled")

  val scalerModel = scaler.fit(featuresForScalingDf)

  val scaledModel = scalerModel.transform(featuresForScalingDf)

  val categorisedDf = categoryColumns.foldLeft(scaledModel) {
    (rDf,col) =>
      val indexer = new StringIndexer()
        .setInputCol(col)
        .setOutputCol(col+"_index")
        .fit(rDf)

      val indexed = indexer.transform(rDf)

      val encoder = new OneHotEncoder()
        .setInputCol(col+"_index")
        .setOutputCol(col+"_vec")

      encoder.transform(indexed)
  }
  val assembler2 = new VectorAssembler()
    .setInputCols(("features_scaled"::categoryColumns.map(_+"_vec")).toArray)
    .setOutputCol("features")

  val readyDataFrame= assembler2.transform(categorisedDf)

  val train = readyDataFrame.select($"features",$"Salary".cast(DoubleType)).sample(withReplacement =  false,0.5,1)
  val test = readyDataFrame.select($"features",$"Salary".cast(DoubleType)).except(train)

  val lr = new LinearRegression()
    .setLabelCol("Salary")
    .setFeaturesCol("features")
    .setPredictionCol("preds")

  val lrModel = lr.fit(train)

  println("Linear MSE = "+lrModel.summary.rootMeanSquaredError)

  val lassoR = new LinearRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(1.0)
    .setLabelCol("Salary")
    .setFeaturesCol("features")
    .setPredictionCol("preds")

  val lassoModel = lassoR.fit(train)

  println("Lasso MSE = "+lassoModel.summary.rootMeanSquaredError)
  val rr = new LinearRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.0)
    .setLabelCol("Salary")
    .setFeaturesCol("features")
    .setPredictionCol("preds")

  val rrModel = rr.fit(train)

  println("Ridge MSE = "+rrModel.summary.rootMeanSquaredError)

}
