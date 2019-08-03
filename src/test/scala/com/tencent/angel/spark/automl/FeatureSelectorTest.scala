package com.tencent.angel.spark.automl

import java.io._

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.feature.operator._
import org.scalatest.FunSuite
import org.scalatest.BeforeAndAfter

class FeatureSelectorTest extends FunSuite with BeforeAndAfter {

  var spark: SparkSession = _
  var numTopFeatures: Int = _

  before {
    spark = SparkSession.builder().master("local").getOrCreate()
    numTopFeatures = 10
  }

  after {
    spark.close()
  }

  test("test_lasso") {
    val data = spark.read.format("libsvm")
      .option("numFeatures", "123")
      .load("data/a9a/a9a_123d_train_trans.libsvm")
      .persist()

    val splitData = data.randomSplit(Array(0.7, 0.3))
    val trainDF = splitData(0)
    val testDF = splitData(1)

    val originalLR = new LogisticRegression()
      .setMaxIter(20)
      .setFeaturesCol("features")

    val originalAUC = originalLR.fit(trainDF).evaluate(testDF).asInstanceOf[BinaryLogisticRegressionSummary].areaUnderROC
    println(s"original feature: auc = $originalAUC")

    val selector = new LassoSelector()
      .setFeaturesCol("features")
      .setOutputCol("selectedFeatures")
      .setNumTopFeatures(numTopFeatures)

    val selectorModel = selector.fit(trainDF)
    deleteRecursively(new File("tmp/model/lasso.selector"))
    selectorModel.save("tmp/model/lasso.selector")
    val load_model = LassoSelectorModel.load("tmp/model/lasso.selector")

    val selectedTrainDF = load_model.transform(trainDF)
    val selectedTestDF = load_model.transform(testDF)

    selectedTestDF.select("selectedFeatures").schema.fields
      .foreach(f => println(f.metadata.toString()))

    val selectedLR = new LogisticRegression()
      .setMaxIter(20)
      .setFeaturesCol("selectedFeatures")

    val selectedAUC = selectedLR.fit(selectedTrainDF).evaluate(selectedTestDF).asInstanceOf[BinaryLogisticRegressionSummary].areaUnderROC
    println(s"selected feature: auc = $selectedAUC")
    spark.close()
  }

  test("test_rf") {
    val data = spark.read.format("libsvm")
      .option("numFeatures", "123")
      .load("data/a9a/a9a_123d_train_trans.libsvm")
      .persist()

    val splitData = data.randomSplit(Array(0.7, 0.3))
    val trainDF = splitData(0)
    val testDF = splitData(1)

    val originalLR = new LogisticRegression()
      .setMaxIter(20)
      .setFeaturesCol("features")

    val originalAUC = originalLR.fit(trainDF).evaluate(testDF).asInstanceOf[BinaryLogisticRegressionSummary].areaUnderROC
    println(s"original feature: auc = $originalAUC")

    val selector = new RandomForestSelector()
      .setFeaturesCol("features")
      .setOutputCol("selectedFeatures")
      .setNumTopFeatures(numTopFeatures)

    val selectorModel = selector.fit(trainDF)
    deleteRecursively(new File("tmp/model/rf.selector"))
    selectorModel.save("tmp/model/rf.selector")
    val load_model = RandomForestSelectorModel.load("tmp/model/rf.selector")

    val selectedTrainDF = load_model.transform(trainDF)
    val selectedTestDF = load_model.transform(testDF)

    selectedTestDF.select("selectedFeatures")
      .schema.fields.foreach(f => println(f.metadata.toString()))

    val selectedLR = new LogisticRegression()
      .setMaxIter(20)
      .setFeaturesCol("selectedFeatures")

    val selectedAUC = selectedLR.fit(selectedTrainDF).evaluate(selectedTestDF).asInstanceOf[BinaryLogisticRegressionSummary].areaUnderROC
    println(s"selected feature: auc = $selectedAUC")
  }

  test("test_variance") {
    val data = spark.read.format("libsvm")
      .option("numFeatures", "123")
      .load("data/a9a/a9a_123d_train_trans.libsvm")
      .persist()

    val splitData = data.randomSplit(Array(0.7, 0.3))
    val trainDF = splitData(0)
    val testDF = splitData(1)

    val originalLR = new LogisticRegression()
      .setMaxIter(20)
      .setFeaturesCol("features")

    val originalAUC = originalLR.fit(trainDF).evaluate(testDF).asInstanceOf[BinaryLogisticRegressionSummary].areaUnderROC
    println(s"original feature: auc = $originalAUC")

    val selector = new VarianceSelector()
      .setFeaturesCol("features")
      .setOutputCol("selectedFeatures")
      .setNumTopFeatures(numTopFeatures)

    val selectorModel = selector.fit(trainDF)
    deleteRecursively(new File("tmp/model/variance.selector"))
    selectorModel.save("tmp/model/variance.selector")
    val load_model = VarianceSelectorModel.load("tmp/model/variance.selector")

    val selectedTrainDF = load_model.transform(trainDF)
    val selectedTestDF = load_model.transform(testDF)

    selectedTestDF.select("selectedFeatures").schema.fields
      .foreach(f => println(f.metadata.toString()))

    val selectedLR = new LogisticRegression()
      .setMaxIter(20)
      .setFeaturesCol("selectedFeatures")

    val selectedAUC = selectedLR.fit(selectedTrainDF).evaluate(selectedTestDF).asInstanceOf[BinaryLogisticRegressionSummary].areaUnderROC
    println(s"selected feature: auc = $selectedAUC")
  }

  test("test_ftest") {
    val data = spark.read.format("libsvm")
      .option("numFeatures", "123")
      .option("vectorType", "dense")
      .load("data/a9a/a9a_123d_train_trans.libsvm")
      .persist()

    val splitData = data.randomSplit(Array(0.7, 0.3))
    val trainDF = splitData(0)
    val testDF = splitData(1)

    val originalLR = new LogisticRegression()
      .setMaxIter(20)
      .setFeaturesCol("features")

    val originalAUC = originalLR.fit(trainDF).evaluate(testDF).asInstanceOf[BinaryLogisticRegressionSummary].areaUnderROC
    println(s"original feature: auc = $originalAUC")

    val selector = new FtestSelector()
      .setFeaturesCol("features")
      .setOutputCol("selectedFeatures")
      .setNumTopFeatures(numTopFeatures)

    val selectorModel = selector.fit(trainDF)
    deleteRecursively(new File("tmp/model/ftest.selector"))
    selectorModel.save("tmp/model/ftest.selector")
    val load_model = FtestSelectorModel.load("tmp/model/ftest.selector")

    val selectedTrainDF = load_model.transform(trainDF)
    val selectedTestDF = load_model.transform(testDF)

    selectedTestDF.select("selectedFeatures").schema.fields
      .foreach(f => println(f.metadata.toString()))

    val selectedLR = new LogisticRegression()
      .setMaxIter(20)
      .setFeaturesCol("selectedFeatures")

    val selectedAUC = selectedLR.fit(selectedTrainDF).evaluate(selectedTestDF).asInstanceOf[BinaryLogisticRegressionSummary].areaUnderROC
    println(s"selected feature: auc = $selectedAUC")
  }

  def deleteRecursively(file: File): Unit = {
    if (file.isDirectory) {
      file.listFiles.foreach(deleteRecursively)
    }
    if (file.exists && !file.delete) {
      throw new Exception(s"Unable to delete ${file.getAbsolutePath}")
    }
  }

}
