package main

import (
	"context"
	"flag"
)

func main() {

	//从启动命令行获取参数并解析
	fileName := *(flag.String("File", "setting.yaml", "path to Config file"))
	flag.Parse()

	//消费者启动
	// 创建一个全局的上下文，用于控制所有后台任务的退出
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // 程序退出时取消

}
