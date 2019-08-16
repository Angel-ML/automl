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
package com.tencent.angel.spark.automl.tuner.kernel

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import com.tencent.angel.spark.automl.tuner.math.SquareDist

/**
  * Square exponential covariance function with isotropic distance measure
  * k(x1, x2) = theta^2 * exp( -(x1-x2)^2 / l^2 )
  * Hyper-parameter: theta is the signal variance, l is the length scale
  **/
case class SquareExpIso() extends Covariance {

  /**
    * the covariance function
    *
    * @param x1
    * @param x2
    * @param params
    * @return
    */
  override def cov(x1: BDM[Double],
                   x2: BDM[Double],
                   params: BDV[Double]): BDM[Double] = {

    require(params.size == 2,
      s"Number of hyper parameters is ${params.length} while expected 2")

    val theta = params(0)
    val l = params(1)

    val distMat = SquareDist(x1, x2)

    val covMatrix = pow(theta, 2) * exp(-0.5 * distMat / pow(l, 2))

    covMatrix
  }

  /**
    * the derivative of covariance function against kernel hyper-parameters
    *
    * @param x1
    * @param x2
    * @param params
    * @return
    */
  override def grad(x1: BDM[Double],
                    x2: BDM[Double],
                    params: BDV[Double]): Array[BDM[Double]] = {

    require(params.size == 2,
      s"Number of hyper parameters is ${params.length} while expected 2")

    val theta = params(0)
    val l = params(1)

    val distMat = SquareDist(x1, x2)
    val r = sqrt(distMat)

    val expDistMat = exp(-0.5 * distMat / pow(l, 2))

    val gradTheta = 2 * theta * expDistMat

    val gradL = pow(theta, 2) * expDistMat *:* distMat / pow(l, 3)

    Array(gradTheta, gradL)
  }
}

