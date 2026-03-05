package main

import (
	"context"
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type Input struct {
	Name string `json:"name" jsonschema:"the name of the person to greet"`
}

type Output struct {
	Greeting string `json:"greeting" jsonschema:"the greeting to tell to the user"`
}

type AddInput struct {
	A int `json:"A" jsonschema:"第一个加数"`
	B int `json:"B" jsonschema:"第二个加数"`
}

type AddOutput struct {
	Result int `json:"result" jsonschema:"计算结果"`
}

func Add(ctx context.Context, req *mcp.CallToolRequest, input AddInput) (
	*mcp.CallToolResult,
	AddOutput,
	error,
) {
	return nil, AddOutput{Result: input.A + input.B}, nil
}

func SayHi(ctx context.Context, req *mcp.CallToolRequest, input Input) (
	*mcp.CallToolResult,
	Output,
	error,
) {
	return nil, Output{Greeting: "Hi " + input.Name}, nil
}

func main() {
	server := mcp.NewServer(&mcp.Implementation{Name: "greeter", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "greet", Description: "say hi"}, SayHi)
	mcp.AddTool(server, &mcp.Tool{Name: "add", Description: "add"}, Add)

	// 正确用法：NewStreamableHTTPHandler，返回 http.Handler
	mcpHandler := mcp.NewStreamableHTTPHandler(func(r *http.Request) *mcp.Server {
		return server
	}, nil)

	router := gin.Default()
	router.Any("/mcp", gin.WrapH(mcpHandler))

	log.Println("MCP server 启动于 :8080")
	if err := router.Run(":8080"); err != nil {
		log.Fatal(err)
	}
}
