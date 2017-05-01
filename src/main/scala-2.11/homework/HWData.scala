package homework

/**
  * Created by Sardor on 4/2/2017.
  */

import java.io.File

import scala.reflect.ClassTag
import breeze.linalg._
import breeze.stats._
import breeze.optimize._
import org.apache.spark
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.rdd.RDD

import scala.io.Source

object HWData {

  val DataDirectory = "data/"
  val fileName = "rep_height_weights.csv"

  def load:HWData =
  {
    val file = Source.fromFile(DataDirectory + fileName)
    val lines = file.getLines.toVector
    val splitLines = lines.map { _.split(',') }

    def fromList[T:ClassTag](index:Int, converter:(String => T)):DenseVector[T] =
      DenseVector.tabulate(lines.size) { irow => converter(splitLines(irow)(index)) }

    val genders = fromList(1, elem => elem.replace("\"", "").head)
    val weights = fromList(2, elem => elem.toDouble)
    val heights = fromList(3, elem => elem.toDouble)
    val reportedWeights = fromList(4, elem => elem.toDouble)
    val reportedHeights = fromList(5, elem => elem.toDouble)

    new HWData(weights, heights, reportedWeights, reportedHeights, genders)
  }

  def loadHouseData(implicit sc:SparkSession):DataFrame = {
    val df = sc.read.option("header","true").option("inferSchema","true").csv(getClass.getResource("hprice1.csv").toString)

    df
  }



}

class HWData(
              val weights:DenseVector[Double],
              val heights:DenseVector[Double],
              val reportedWeights:DenseVector[Double],
              val reportedHeights:DenseVector[Double],
              val genders:DenseVector[Char]
            ) {

  val npoints = heights.length
  require(weights.length == npoints)
  require(reportedWeights.length == npoints)
  require(genders.length == npoints)
  require(reportedHeights.length == npoints)

  lazy val rescaledHeights:DenseVector[Double] =
    (heights - mean(heights)) / stddev(heights)

  lazy val rescaledWeights:DenseVector[Double] =
    (weights - mean(weights)) / stddev(weights)

  lazy val featureMatrix:DenseMatrix[Double] =
    DenseMatrix.horzcat(
      DenseMatrix.ones[Double](npoints, 1),
      rescaledHeights.toDenseMatrix.t,
      rescaledWeights.toDenseMatrix.t
    )

  lazy val target:DenseVector[Double] =
    genders.values.map { gender => if(gender == 'M') 1.0 else 0.0 }

  override def toString:String = s"HWData [ $npoints rows ]"

}
