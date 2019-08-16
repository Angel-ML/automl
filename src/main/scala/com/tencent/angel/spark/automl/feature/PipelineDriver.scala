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
package com.tencent.angel.spark.automl.feature

import com.tencent.angel.spark.automl.feature.preprocess.{MinMaxScalerWrapper, StandardScalerWrapper}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object PipelineDriver {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local").getOrCreate()

    //    val inputDF = spark.createDataFrame(Seq(
    //      (0L, "a b c d e spark", 1.0),
    //      (1L, "b d", 0.0),
    //      (2L, "spark f g h", 1.0),
    //      (3L, "hadoop mapreduce", 0.0)
    //    )).toDF("id", "text", "label")

    val inputDF = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -1.0)),
      (1, Vectors.dense(2.0, 1.1, 1.0)),
      (2, Vectors.dense(3.0, 10.1, 3.0))
    )).toDF("id", "numerical")

    val pipelineWrapper = new PipelineWrapper()

    val transformers = Array[TransformerWrapper](
      new MinMaxScalerWrapper(),
      new StandardScalerWrapper()
    )

    val stages = PipelineBuilder.build(transformers)

    print(transformers(0).getInputCols)

    pipelineWrapper.setStages(stages)

    val model: PipelineModelWrapper = pipelineWrapper.fit(inputDF)

    val outDF = model.transform(inputDF)

    outDF.show()

  }

}
