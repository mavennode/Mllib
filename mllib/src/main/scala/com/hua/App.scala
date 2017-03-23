package com.hua

import scala.util.Random

/**
 * Hello world!
 *
 */
object App {
  def main(args: Array[String]): Unit = {
    val data = Random.nextDouble()
    val t = ("13056677799","张三","13056677799","woman","四川","地推注册邀请",5,"2015-10-11 11:32:19",Some("delete"))
    println(t._2._1)
  }
}
