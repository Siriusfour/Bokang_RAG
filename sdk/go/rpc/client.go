package rpc

import (
	"context"
	"errors"
	"strings"
	"time"

	"rag/sdk/rpc/pb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type Client struct {
	conn   *grpc.ClientConn
	client pb.AskServiceClient
}

type Option func(*clientOptions)

type clientOptions struct {
	address     string
	timeout     time.Duration
	dialOptions []grpc.DialOption
}

func WithAddress(addr string) Option {
	return func(o *clientOptions) {
		if strings.TrimSpace(addr) != "" {
			o.address = addr
		}
	}
}

func WithTimeout(timeout time.Duration) Option {
	return func(o *clientOptions) {
		if timeout > 0 {
			o.timeout = timeout
		}
	}
}

func WithDialOption(opt grpc.DialOption) Option {
	return func(o *clientOptions) {
		if opt != nil {
			o.dialOptions = append(o.dialOptions, opt)
		}
	}
}

// 新建客户端
func NewClient(opts ...Option) (*Client, error) {
	options := &clientOptions{
		address: "127.0.0.1:7071",
		timeout: 15 * time.Second,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(options)
		}
	}
	if len(options.dialOptions) == 0 {
		options.dialOptions = []grpc.DialOption{
			grpc.WithTransportCredentials(insecure.NewCredentials()),
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), options.timeout)
	defer cancel()
	conn, err := grpc.DialContext(ctx, options.address, options.dialOptions...)
	if err != nil {
		return nil, err
	}
	return &Client{
		conn:   conn,
		client: pb.NewAskServiceClient(conn),
	}, nil
}

func (c *Client) Close() error {
	if c == nil || c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

func (c *Client) Ask(ctx context.Context, question string) (string, error) {
	return c.AskWithSession(ctx, question, "")
}

func (c *Client) AskWithSession(ctx context.Context, question string, sessionId string) (string, error) {
	if c == nil || c.client == nil {
		return "", errors.New("client is nil")
	}
	if strings.TrimSpace(question) == "" {
		return "", errors.New("question is required")
	}
	req := &pb.AskRequest{
		Question:  question,
		SessionId: sessionId,
	}
	resp, err := c.client.Ask(ctx, req)
	if err != nil {
		return "", err
	}
	return resp.GetAnswer(), nil
}
