/*
 * Tencent is pleased to support the open source community by making Angel available.
 *
 * Copyright (C) 2017-2018 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */

package com.tencent.angel.spark.automl

import java.io.File

import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.operator._
import org.apache.spark.sql.SparkSession
import org.scalatest.FunSuite
import org.scalatest.BeforeAndAfter

import scala.collection.mutable.ArrayBuffer

class FeatureEngineeringTest extends FunSuite with BeforeAndAfter {

  var spark: SparkSession = _

  before {
    spark = SparkSession.builder().master("local").getOrCreate()
  }

  after {
    spark.close()
  }

 test("test_iterative_cross") {

    val dim = 123
    val incDim = 10
    val iter = 2
    val modelPath = "tmp/model/feature_engineer"

    val data = spark.read.format("libsvm")
      .option("numFeatures", dim)
      .load("data/a9a/a9a_123d_train_trans.libsvm")
      .persist()

    val featureMap: Map[Int, Int] = Map[Int, Int]()

    val pipelineStages: ArrayBuffer[PipelineStage] = new ArrayBuffer
    val usedFields: ArrayBuffer[String] = new ArrayBuffer[String]()

    val cartesianPrefix = "_f"
    val selectorPrefix = "_select"
    val filterPrefix = "_filter"
    var curField = "features"
    usedFields += curField

    (0 until iter).foreach { iter =>
      // add cartesian operator
      val cartesian = new VectorCartesian()
      .setInputCols(Array("features", curField))
      .setOutputCol(curField + cartesianPrefix)
      println(s"Cartesian -> input features and $curField, output ${curField + cartesianPrefix}")
      pipelineStages += cartesian
      curField += cartesianPrefix

      // add selector operator
      val selector = new RandomForestSelector()
        .setFeaturesCol(curField)
        .setLabelCol("label")
        .setOutputCol(curField + selectorPrefix)
        .setNumTopFeatures(incDim)
      println(s"Selector -> input $curField, output ${curField + selectorPrefix}")
      pipelineStages += selector
      curField += selectorPrefix

      // add filter operator
      val filter = new VectorFilterZero(featureMap)
        .setInputCol(curField)
        .setOutputCol(curField + filterPrefix)
      println(s"Filter -> input $curField, output ${curField + filterPrefix}")
      pipelineStages += filter
      curField += filterPrefix
      usedFields += curField
    }

    println(s"used fields: ${usedFields.toArray.mkString(",")}")

    val assembler = new VectorAssembler()
      .setInputCols(usedFields.toArray)
      .setOutputCol("assembled_features")
    pipelineStages += assembler

    val pipeline = new Pipeline()
      .setStages(pipelineStages.toArray)

    val model = pipeline.fit(data)
    deleteRecursively(new File(modelPath))
    model.save(modelPath)
    val load_model = PipelineModel.load(modelPath)

    val crossDF = load_model.transform(data).persist()
    data.unpersist()
    crossDF.show(1)

    usedFields.takeRight(usedFields.length - 1).foreach{ field =>
      println(crossDF.select(field).schema.fields.last.metadata
        .getStringArray(MetadataTransformUtils.DERIVATION).length + " cross features in " + usedFields.last)
      println(crossDF.select(field).schema.fields.last.metadata
        .getStringArray(MetadataTransformUtils.DERIVATION).mkString(","))
    }

    val splitDF = crossDF.randomSplit(Array(0.7, 0.3))

    val trainDF = splitDF(0).persist()
    val testDF = splitDF(1).persist()

    val originalLR = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setMaxIter(100)
      .setRegParam(0.01)
    val originalAUC = originalLR.fit(trainDF).evaluate(testDF)
      .asInstanceOf[BinaryLogisticRegressionSummary].areaUnderROC
    println(s"original features auc = $originalAUC")

    val crossLR = new LogisticRegression()
      .setFeaturesCol("assembled_features")
      .setLabelCol("label")
      .setMaxIter(100)
      .setRegParam(0.01)
    val crossAUC = crossLR.fit(trainDF).evaluate(testDF)
      .asInstanceOf[BinaryLogisticRegressionSummary].areaUnderROC
    println(s"cross features auc = $crossAUC")
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
