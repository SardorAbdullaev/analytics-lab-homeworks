package homework

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.log4j._
import org.apache.spark
import org.apache.spark.sql.SparkSession



/**
  * Created by Sardor on 3/31/2017.
  */
object Main {
  //Logger.getLogger("org").setLevel(Level.ERROR)
  def main(args: Array[String]): Unit = {

    //val conf = new SparkConf().setMaster("local").setAppName("DS_Homework")
    //val sc = new SparkContext(conf)
    implicit val sparkSession = SparkSession.builder.
      master("local")
      .appName("spark session example")
      .getOrCreate()

    val df = HWData.loadHouseData
    df.head(5) // returns an Array
    println("\n")
    for(line <- df.head(10)){
      println(line)
    }
    /*val input = sc.textFile("C:/spark-2.1.0-bin-hadoop2.7/README.md")
      val words = input.flatMap(line => line.split(" "))
      val counts = words.map(word => (word,1)).reduceByKey{
        case (x,y)=> x + y
      }
      counts.saveAsTextFile("C:/outputfile")
*/
//    val data = Array(1, 2, 3, 4, 5)
//    val distData: RDD[Int] = sc.parallelize(data)

  }
}
