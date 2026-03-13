package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	baseURL    string
	httpClient *http.Client
	headers    map[string]string
}

type Option func(*Client)

func WithHTTPClient(client *http.Client) Option {
	return func(c *Client) {
		if client != nil {
			c.httpClient = client
		}
	}
}

func WithHeader(key, value string) Option {
	return func(c *Client) {
		if key == "" {
			return
		}
		c.headers[key] = value
	}
}

func NewClient(baseURL string, opts ...Option) *Client {
	baseURL = strings.TrimRight(baseURL, "/")
	if baseURL == "" {
		baseURL = "http://127.0.0.1:7070"
	}
	client := &Client{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
		headers: map[string]string{},
	}
	for _, opt := range opts {
		if opt != nil {
			opt(client)
		}
	}
	return client
}

func (c *Client) Ask(ctx context.Context, req AskRequest) (*AskResponse, error) {
	if strings.TrimSpace(req.Question) == "" {
		return nil, errors.New("question is required")
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	url := c.baseURL + "/ask"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("content-type", "application/json")
	for k, v := range c.headers {
		if k != "" {
			httpReq.Header.Set(k, v)
		}
	}
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return nil, fmt.Errorf("ask failed: %d: %s", resp.StatusCode, string(respBody))
	}
	var out AskResponse
	if err := json.Unmarshal(respBody, &out); err != nil {
		return nil, err
	}
	return &out, nil
}
