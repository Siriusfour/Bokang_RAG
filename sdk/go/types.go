package rag

type AskRequest struct {
	Question string `json:"question"`
	ThreadID string `json:"threadId,omitempty"`
}

type AskResponse struct {
	Answer  string      `json:"answer"`
	Context interface{} `json:"context"`
	State   interface{} `json:"state"`
}
