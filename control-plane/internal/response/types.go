package response

type Response struct {
	Metadata ResponseMetadata
	Payload  GenerateResponse
}

type ResponseMetadata struct {
}

type GenerateResponse struct {
	RequestID       *string `json:"request_id"`
	Text            string  `json:"text"`
	GeneratedTokens int     `json:"generated_tokens"`
}
