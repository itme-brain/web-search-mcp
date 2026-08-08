# Evidence resources

Every successfully scraped search document is stored in shared Valkey under a
content-derived SHA-256 identifier. Its chunks receive identifiers derived from
the document ID, position, and text. These stable handles are exposed as:

- `web-search://documents/{document_id}`
- `web-search://chunks/{chunk_id}`

Search passages include their chunk URI when available, and search results
include the complete document URI. Clients can use the `read_evidence` tool or
MCP resource reads to expand either handle without fetching the web page again.
The payload expires according to `EVIDENCE_TTL_S` and can be resolved by any MCP
replica connected to the same Valkey instance.

The public search response remains compact: passages are not copied into legacy
brief/finding/evidence fields, and complete document content stays behind its
resource handle.
