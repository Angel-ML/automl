package com.tencent.angel.spark.automl

import java.io._

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.feature.operator._
import org.junit.Test

class FeatureSelectorTest {

  val spark = SparkSession.builder().master("local").getOrCreate()

  val numTopFeatures = 10

  @Test def testLasso(): Unit = {
    System.setOut(new PrintStream(new FileOutputStream("tmp/log/lasso.selector.out")))
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

    spark.stop()
  }

  @Test
  def testRandomForest(): Unit = {
    System.setOut(new PrintStream(new FileOutputStream("tmp/log/rf.selector.out")))
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

    spark.stop()
  }

  @Test
  def testVariance(): Unit = {
    System.setOut(new PrintStream(new FileOutputStream("tmp/log/variance.selector.out")))
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

    spark.stop()
  }

  @Test
  def testFtest(): Unit = {
    System.setOut(new PrintStream(new FileOutputStream("tmp/log/ftest.selector.out")))
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

    spark.stop()
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
