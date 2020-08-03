import tech.tablesaw.api.{DoubleColumn, IntColumn, StringColumn, Table}
import tech.tablesaw.api._
import tech.tablesaw.io.csv.CsvReadOptions

import scala.collection.JavaConverters._
import util.control.Breaks._
import tech.tablesaw.aggregate.AggregateFunctions._

import scala.collection.mutable.ArrayBuffer


object testclass {
  def main(args: Array[String]): Unit = {
    val readOptions = CsvReadOptions.builder("../../../zwpweka/experiments/creditg/out/singleTechnique30.csv").header(false).build()
    val table = Table.read().usingOptions(readOptions)
    // set names
    val columnNames = List("identifier", "dataset", "algTypeString", "algName", "optRd", "i", "score", "subScore", "time", "argStr")
    for (i <- columnNames.indices) {
      table.column(i).setName(columnNames(i))
    }

    val cleanDf = table.select("identifier", "dataset", "algTypeString", "algName", "optRd", "i", "score").dropDuplicateRows().select("algName", "algTypeString", "optRd", "i", "score")

    // discard first round
    //    algorithmList = cleanDf[lambda df: df.optRd == optRd][['algTypeString', 'algName']].drop_duplicates()
    //    history = pd.DataFrame(columns=['algName', 'algTypeString', 'optRd', 'score'])

    def appendHistory(history: Table, algName: String, algTypeString: String, optRd: Int, score: Double): Table = {
      val row = history.appendRow()
      row.setString("algName", "asf")
      row.setString("algTypeString", "aasf")
      row.setInt("optRd", 1)
      row.setDouble("score", 12)
      history
    }

    val optRd = 0
    val algTypeString = "Base"
    val algorithmToKeep = 9

    val algorithmList = cleanDf.where(cleanDf.intColumn("optRd").isEqualTo(optRd)).select("algTypeString", "algName").dropDuplicateRows()
    var history = Table.create("history").addColumns(
      StringColumn.create("algName"),
      StringColumn.create("algTypeString"),
      IntColumn.create("optRd"),
      DoubleColumn.create("score")
    )

    val algoList = algorithmList.where(algorithmList.stringColumn("algTypeString").isEqualTo(algTypeString)).dropDuplicateRows().column("algName").asList().asScala
    for ((algorithm: String, i) <- algoList.view.zipWithIndex) {
      var significant = false
      var skippedCount = 0
      for ((score: Double, j) <- cleanDf.where(cleanDf.intColumn("optRd").isEqualTo(optRd)).where(cleanDf.stringColumn("algName").isEqualTo(algorithm)).column("score").asList().asScala.view.zipWithIndex) {
        breakable {
          if (significant) {
            skippedCount += 1
            if (j == cleanDf.where(cleanDf.intColumn("optRd").isEqualTo(optRd)).where(cleanDf.stringColumn("algName").isEqualTo(algorithm)).column("score").size() - 1 & skippedCount > 0) {
              println("skipped", algorithm)
              println("times", skippedCount)
            }
            break() // continue, means braking the breakable and continue to the next iteration
          }
          history = appendHistory(history = history, algName = algorithm, algTypeString = algTypeString, optRd = optRd, score = score)
          if (i > algorithmList.where(algorithmList.stringColumn("algTypeString").isEqualTo(algTypeString)).column("algName").size() * 0.5) {
            val avgRunNumber = history.where(history.stringColumn("algName").isNotEqualTo(algorithm)).summarize("algName", count).apply().doubleColumn("Count [algName]").mean()
            val currentRunNumber = history.where(history.stringColumn("algName").isEqualTo(algorithm)).rowCount()
            if (currentRunNumber > 0.5 * avgRunNumber) {
              if (List("weka.classifiers.trees.RandomForest", "weka.classifiers.functions.SMO").contains(algorithm)) {
                break()
              }
            }
            // make history better
            // early stop 2
            // early stop 3
          }
        }
      }

    }

    //    cleanDf.where(cleanDf.stringColumn("algName").isNotEqualTo(algorithm)).select("algName", "score").summarize("score", min).by("algName")

    // early stop 3
    def earlyStop3(algorithm: String, history: Table, optRd: Int): (Boolean, Table) = {
      //      historicalBest = history[lambda df: df.algName != algorithm][['algName', 'score']].groupby('algName',
      //      as_index=False).min().score
      val historicalBest = history.where(history.stringColumn("algName").isNotEqualTo(algorithm)).select("algName", "score").summarize("score", min).by("algName").doubleColumn("Min [score]")
      val currentBest = history.where(history.stringColumn("algName").isEqualTo(algorithm)).doubleColumn("score").min()
      var mean_list = List[Double]()
      mean_list = 1.0 :: mean_list
      for (i <- 1 to 10000) {
        var a_resample = List[Double]()
        for (j <- 1 to historicalBest.size()) {
          a_resample = historicalBest.sampleN(1) :: a_resample

          val col = DoubleColumn.create("col")
          a_resample.foreach(i => col.append(i))
          mean_list = col.mean() :: mean_list
        }
        mean_list = currentBest :: mean_list
      }
      val percentile = mean_list.sorted.indexOf(currentBest) / mean_list.size
      val significant = percentile > 0.95
      var returnHistory = history
      if (significant) {
        returnHistory = history.where(history.stringColumn("algName").isNotEqualTo(algorithm))
      }
      (significant, history)
    }



    history
  }
  val readOptions = CsvReadOptions.builder("../../../zwpweka/experiments/creditg/out/earlyStoppedAlgorithm.csv").header(false).build()
  val table = Table.read().usingOptions(readOptions)
  // set names
  val columnNames = List("algName", "optRd", "i", "status")
  for (i <- columnNames.indices) {
    table.column(i).setName(columnNames(i))
  }
  table
  val optRd = 0
  val algType = "base"
  table.where(table.intColumn("optRd").isEqualTo(0)).stringColumn("algName")


}
