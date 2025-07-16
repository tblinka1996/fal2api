import express from 'express';
import { fal } from '@fal-ai/client';

// --- Key Management Setup ---
// Read comma-separated keys from the SINGLE environment variable FAL_KEY
const FAL_KEY_STRING = process.env.FAL_KEY;
const API_KEY = process.env.API_KEY; // Custom API Key for proxy auth

if (!FAL_KEY_STRING) {
    console.error("Error: FAL_KEY environment variable is not set.");
    console.error("Ensure FAL_KEY contains a comma-separated list of your Fal AI keys.");
    process.exit(1);
}

// Parse the comma-separated keys from FAL_KEY_STRING
const falKeys = FAL_KEY_STRING.split(',')
    .map(key => key.trim()) // Remove leading/trailing whitespace
    .filter(key => key.length > 0); // Remove any empty strings resulting from extra commas

if (falKeys.length === 0) {
    console.error("Error: No valid FAL keys found in the FAL_KEY environment variable after parsing.");
    console.error("Ensure FAL_KEY is a comma-separated list, e.g., 'key1,key2,key3'.");
    process.exit(1);
}

if (!API_KEY) {
    console.error("Error: API_KEY environment variable is not set.");
    process.exit(1);
}

let currentKeyIndex = 0;
const invalidKeys = new Set(); // Keep track of keys that failed

console.log(`Loaded ${falKeys.length} Fal AI Key(s) from the FAL_KEY environment variable.`);

// Function to get the next valid key in a round-robin fashion
function getNextValidKey() {
    if (invalidKeys.size >= falKeys.length) {
        console.error("All Fal AI keys are marked as invalid.");
        return null; // No valid keys left
    }

    const initialIndex = currentKeyIndex;
    let attempts = 0;
    while (attempts < falKeys.length) {
        const keyIndex = currentKeyIndex % falKeys.length;
        const key = falKeys[keyIndex];

        // Move to the next index for the *next* call
        currentKeyIndex = (keyIndex + 1) % falKeys.length;

        if (!invalidKeys.has(key)) {
            // Found a valid key
            console.log(`Using Fal Key index: ${keyIndex} (from FAL_KEY list)`);
            return { key, index: keyIndex };
        }

        attempts++;
        // Continue loop to check the next key
    }

    // Should not be reached if invalidKeys.size check is correct, but as a safeguard
    console.error("Could not find a valid Fal AI key after checking all.");
    return null;
}

// Function to check if an error is likely related to a bad key
// NOTE: This is a heuristic. You might need to adjust based on actual errors from Fal AI.
function isKeyRelatedError(error) {
    const message = error?.message?.toLowerCase() || '';
    const status = error?.status; // Check if the error object has a status code

    // Check for specific HTTP status codes indicative of auth/permission issues
    if (status === 401 || status === 403) {
        console.warn(`Detected potential key-related error (HTTP Status: ${status}).`);
        return true;
    }
    // Check for common error message patterns
    if (message.includes('invalid api key') ||
        message.includes('authentication failed') ||
        message.includes('permission denied') ||
        message.includes('quota exceeded') || // Include quota errors as key-related for rotation
        message.includes('forbidden') ||
        message.includes('unauthorized')) { // Add 'unauthorized'
        console.warn(`Detected potential key-related error (message: ${message})`);
        return true;
    }
    // Add more specific checks based on observed Fal AI errors if needed
    return false;
}
// --- End Key Management Setup ---

const app = express();
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

const PORT = process.env.PORT || 3000;

// API Key 鉴权中间件 (unchanged)
const apiKeyAuth = (req, res, next) => {
    const authHeader = req.headers['authorization'];

    if (!authHeader) {
        console.warn('Unauthorized: No Authorization header provided');
        return res.status(401).json({ error: 'Unauthorized: No API Key provided' });
    }

    const authParts = authHeader.split(' ');
    if (authParts.length !== 2 || authParts[0].toLowerCase() !== 'bearer') {
        console.warn('Unauthorized: Invalid Authorization header format');
        return res.status(401).json({ error: 'Unauthorized: Invalid Authorization header format' });
    }

    const providedKey = authParts[1];
    if (providedKey !== API_KEY) {
        console.warn('Unauthorized: Invalid API Key');
        return res.status(401).json({ error: 'Unauthorized: Invalid API Key' });
    }

    next();
};

app.use(['/v1/models', '/v1/chat/completions'], apiKeyAuth);

// === 全局定义限制 === (unchanged)
const PROMPT_LIMIT = 4800;
const SYSTEM_PROMPT_LIMIT = 4800;
// === 限制定义结束 ===

// 定义 fal-ai/any-llm 支持的模型列表 (unchanged)
const FAL_SUPPORTED_MODELS = [
    "anthropic/claude-sonnet-4",
    "anthropic/claude-opus-4",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-5-haiku",
    "anthropic/claude-3-haiku",
    "google/gemini-pro-1.5",
    "google/gemini-flash-1.5",
    "google/gemini-flash-1.5-8b",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.2-1b-instruct",
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "deepseek/deepseek-r1",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-scout"
];

// Helper function getOwner (unchanged)
const getOwner = (modelId) => {
    if (modelId && modelId.includes('/')) {
        return modelId.split('/')[0];
    }
    return 'fal-ai';
}

// GET /v1/models endpoint (unchanged)
app.get('/v1/models', (req, res) => {
    console.log("Received request for GET /v1/models");
    try {
        const modelsData = FAL_SUPPORTED_MODELS.map(modelId => ({
            id: modelId, object: "model", created: Math.floor(Date.now() / 1000), owned_by: getOwner(modelId) // Use current time for created
        }));
        res.json({ object: "list", data: modelsData });
        console.log("Successfully returned model list.");
    } catch (error) {
        console.error("Error processing GET /v1/models:", error);
        res.status(500).json({ error: "Failed to retrieve model list." });
    }
});

// convertMessagesToFalPrompt 函数 (unchanged)
function convertMessagesToFalPrompt(messages) {
    // ... (keep existing conversion logic)
    let fixed_system_prompt_content = "";
    const conversation_message_blocks = [];
    // console.log(`Original messages count: ${messages.length}`); // Less verbose logging

    // 1. 分离 System 消息，格式化 User/Assistant 消息
    for (const message of messages) {
        let content = (message.content === null || message.content === undefined) ? "" : String(message.content);
        switch (message.role) {
            case 'system':
                fixed_system_prompt_content += `System: ${content}\n\n`;
                break;
            case 'user':
                conversation_message_blocks.push(`Human: ${content}\n\n`);
                break;
            case 'assistant':
                conversation_message_blocks.push(`Assistant: ${content}\n\n`);
                break;
            default:
                console.warn(`Unsupported role: ${message.role}`);
                continue;
        }
    }

    // 2. 截断合并后的 system 消息（如果超长）
    if (fixed_system_prompt_content.length > SYSTEM_PROMPT_LIMIT) {
        const originalLength = fixed_system_prompt_content.length;
        fixed_system_prompt_content = fixed_system_prompt_content.substring(0, SYSTEM_PROMPT_LIMIT);
        console.warn(`Combined system messages truncated from ${originalLength} to ${SYSTEM_PROMPT_LIMIT}`);
    }
    fixed_system_prompt_content = fixed_system_prompt_content.trim();

    // 3. 计算 system_prompt 中留给对话历史的剩余空间
    let space_occupied_by_fixed_system = 0;
    if (fixed_system_prompt_content.length > 0) {
         space_occupied_by_fixed_system = fixed_system_prompt_content.length + 4; // 预留 \n\n...\n\n 的长度
    }
     const remaining_system_limit = Math.max(0, SYSTEM_PROMPT_LIMIT - space_occupied_by_fixed_system);
    // console.log(`Trimmed fixed system prompt length: ${fixed_system_prompt_content.length}. Approx remaining system history limit: ${remaining_system_limit}`);

    // 4. 反向填充 User/Assistant 对话历史
    const prompt_history_blocks = [];
    const system_prompt_history_blocks = [];
    let current_prompt_length = 0;
    let current_system_history_length = 0;
    let promptFull = false;
    let systemHistoryFull = (remaining_system_limit <= 0);

    // console.log(`Processing ${conversation_message_blocks.length} user/assistant messages for recency filling.`);
    for (let i = conversation_message_blocks.length - 1; i >= 0; i--) {
        const message_block = conversation_message_blocks[i];
        const block_length = message_block.length;

        if (promptFull && systemHistoryFull) {
            // console.log(`Both prompt and system history slots full. Omitting older messages from index ${i}.`);
            break;
        }

        if (!promptFull) {
            if (current_prompt_length + block_length <= PROMPT_LIMIT) {
                prompt_history_blocks.unshift(message_block);
                current_prompt_length += block_length;
                continue;
            } else {
                promptFull = true;
                // console.log(`Prompt limit (${PROMPT_LIMIT}) reached. Trying system history slot.`);
            }
        }

        if (!systemHistoryFull) {
            if (current_system_history_length + block_length <= remaining_system_limit) {
                 system_prompt_history_blocks.unshift(message_block);
                 current_system_history_length += block_length;
                 continue;
            } else {
                 systemHistoryFull = true;
                 // console.log(`System history limit (${remaining_system_limit}) reached.`);
            }
        }
    }

    // 5. 组合最终的 prompt 和 system_prompt
    const system_prompt_history_content = system_prompt_history_blocks.join('').trim();
    const final_prompt = prompt_history_blocks.join('').trim();
    const SEPARATOR = "\n\n-------下面是比较早之前的对话内容-----\n\n";
    let final_system_prompt = "";
    const hasFixedSystem = fixed_system_prompt_content.length > 0;
    const hasSystemHistory = system_prompt_history_content.length > 0;

    if (hasFixedSystem && hasSystemHistory) {
        final_system_prompt = fixed_system_prompt_content + SEPARATOR + system_prompt_history_content;
        // console.log("Combining fixed system prompt and history with separator.");
    } else if (hasFixedSystem) {
        final_system_prompt = fixed_system_prompt_content;
        // console.log("Using only fixed system prompt.");
    } else if (hasSystemHistory) {
        final_system_prompt = system_prompt_history_content;
        // console.log("Using only history in system prompt slot.");
    }

    const result = {
        system_prompt: final_system_prompt,
        prompt: final_prompt
    };

    // console.log(`Final system_prompt length (Sys+Separator+Hist): ${result.system_prompt.length}`);
    // console.log(`Final prompt length (Hist): ${result.prompt.length}`);

    return result;
}
// === convertMessagesToFalPrompt 函数结束 ===


// --- Helper function to make Fal AI request with retries ---
async function makeFalRequestWithRetry(falInput, stream = false) {
    let attempts = 0;
    const maxAttempts = falKeys.length; // Try each key at most once per request
    const attemptedKeysInThisRequest = new Set(); // Track keys tried for *this* specific request

    while (attempts < maxAttempts) {
        const keyInfo = getNextValidKey();

        if (!keyInfo) {
            // This happens if all keys are currently in the invalidKeys set
            throw new Error("No valid Fal AI keys available (all marked as invalid).");
        }

        // Avoid retrying the *exact same key* within the *same request attempt cycle*
        // This guards against potential infinite loops if getNextValidKey had issues
        if (attemptedKeysInThisRequest.has(keyInfo.key)) {
             console.warn(`Key at index ${keyInfo.index} already attempted for this request cycle. Skipping.`);
             // Don't increment attempts here, as we didn't actually *use* the key.
             // Let the loop continue to find the next *different* valid key.
             // If all keys are invalid, the check at the start of the loop handles it.
             continue;
        }
        attemptedKeysInThisRequest.add(keyInfo.key);
        attempts++; // Count this as a distinct attempt with a key

        try {
            console.log(`Attempt ${attempts}/${maxAttempts}: Trying Fal Key index ${keyInfo.index}...`);

            // *** CRITICAL: Reconfigure fal client with the selected key ***
            console.warn("Concurrency Warning: Reconfiguring global fal client. Ensure sufficient instance isolation if under high load.");
            fal.config({ credentials: keyInfo.key });

            if (stream) {
                // Return the stream directly for the caller to handle
                const falStream = await fal.stream("fal-ai/any-llm", { input: falInput });
                console.log(`Successfully initiated stream with key index ${keyInfo.index}.`);
                return falStream; // Success, let the caller handle iteration
            } else {
                // For non-stream, wait for the result here
                console.log(`Executing non-stream request with key index ${keyInfo.index}...`);
                const result = await fal.subscribe("fal-ai/any-llm", { input: falInput, logs: true });
                console.log(`Successfully received non-stream result with key index ${keyInfo.index}.`);

                // Check for errors *within* the successful response structure
                 if (result && result.error) {
                     console.error(`Fal-ai returned an error in non-stream result (Key Index ${keyInfo.index}):`, result.error);
                     // Treat this like a general Fal error, not necessarily a key error unless message indicates it
                     // Convert it to a standard Error object to be caught below
                     throw new Error(`Fal-ai error in result: ${JSON.stringify(result.error)}`);
                 }
                return result; // Success
            }
        } catch (error) {
            console.error(`Error using Fal Key index ${keyInfo.index}:`, error.message || error);

            if (isKeyRelatedError(error)) {
                console.warn(`Marking Fal Key index ${keyInfo.index} as invalid due to error.`);
                invalidKeys.add(keyInfo.key);
                // Continue to the next iteration to try another key
            } else {
                // Not identified as a key-related error (e.g., network issue, bad input, internal Fal error)
                // Fail the request immediately, don't retry with other keys for this type of error.
                console.error("Error does not appear to be key-related. Failing request without further retries.");
                throw error; // Re-throw the original error to be caught by the main handler
            }
        }
    }

    // If the loop finishes, it means all keys were tried and marked invalid *within this request cycle*
    throw new Error(`Request failed after trying ${attempts} unique Fal key(s). All failed with key-related errors or were already marked invalid.`);
}


// POST /v1/chat/completions endpoint (Modified to use retry logic)
app.post('/v1/chat/completions', async (req, res) => {
    const { model, messages, stream = false, reasoning = false, ...restOpenAIParams } = req.body;

    // Basic logging for request entry
    console.log(`--> POST /v1/chat/completions | Model: ${model} | Stream: ${stream}`);

    if (!FAL_SUPPORTED_MODELS.includes(model)) {
         console.warn(`Warning: Requested model '${model}' is not in the explicitly supported list. Proxy will still attempt.`);
    }
    if (!model || !messages || !Array.isArray(messages) || messages.length === 0) {
        console.error("Invalid request: Missing 'model' or 'messages' array.");
        return res.status(400).json({ error: 'Missing or invalid parameters: model and messages array are required.' });
    }

    try {
        // --- Prepare Input ---
        const { prompt, system_prompt } = convertMessagesToFalPrompt(messages);
        const falInput = {
            model: model,
            prompt: prompt,
            ...(system_prompt && { system_prompt: system_prompt }),
            reasoning: !!reasoning, // Ensure boolean
        };
        // console.log("Fal Input:", JSON.stringify(falInput, null, 2)); // Verbose logging
        console.log("Attempting Fal request with key rotation/retry...");

        // --- Handle Stream vs Non-Stream ---
        if (stream) {
            res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');
            res.setHeader('Access-Control-Allow-Origin', '*'); // Consider restricting in production
            res.flushHeaders();

            let previousOutput = '';
            let falStream;

            try {
                 // Initiate stream using the retry logic
                 falStream = await makeFalRequestWithRetry(falInput, true);

                 // Process the stream events
                for await (const event of falStream) {
                    const currentOutput = (event && typeof event.output === 'string') ? event.output : '';
                    const isPartial = (event && typeof event.partial === 'boolean') ? event.partial : true;
                    const errorInfo = (event && event.error) ? event.error : null;

                     if (errorInfo) {
                        // Log error from within the stream, but continue processing if possible
                        console.error("Error received *within* fal stream event:", errorInfo);
                        // Send an error chunk to the client (optional, depends on desired behavior)
                        const errorChunk = { id: `chatcmpl-${Date.now()}-error`, object: "chat.completion.chunk", created: Math.floor(Date.now() / 1000), model: model, choices: [{ index: 0, delta: {}, finish_reason: "error", message: { role: 'assistant', content: `Fal Stream Event Error: ${JSON.stringify(errorInfo)}` } }] };
                        // Safety check before writing
                        if (!res.writableEnded) {
                             res.write(`data: ${JSON.stringify(errorChunk)}\n\n`);
                        } else {
                             console.warn("Stream already ended when trying to write stream event error.");
                        }
                        // Decide whether to break or continue based on error severity if needed
                    }

                    // Calculate delta (same logic as before)
                    let deltaContent = '';
                    if (currentOutput.startsWith(previousOutput)) {
                        deltaContent = currentOutput.substring(previousOutput.length);
                    } else if (currentOutput.length > 0) {
                         // console.warn("Fal stream output mismatch. Sending full current output as delta."); // Less verbose
                         deltaContent = currentOutput;
                         previousOutput = ''; // Reset previous output on mismatch
                    }
                    previousOutput = currentOutput;

                    // Send OpenAI compatible chunk
                    if (deltaContent || !isPartial) { // Send even if delta is empty when finishing
                        const openAIChunk = {
                            id: `chatcmpl-${Date.now()}`, // Consider more unique ID if needed
                            object: "chat.completion.chunk",
                            created: Math.floor(Date.now() / 1000),
                            model: model, // Echo back the requested model
                            choices: [{
                                index: 0,
                                delta: { content: deltaContent },
                                finish_reason: isPartial === false ? "stop" : null
                             }]
                        };
                        // Safety check before writing
                        if (!res.writableEnded) {
                            res.write(`data: ${JSON.stringify(openAIChunk)}\n\n`);
                        } else {
                             console.warn("Stream already ended when trying to write data chunk.");
                        }
                    }
                 } // End for-await loop

                 // Send the final [DONE] marker
                 if (!res.writableEnded) {
                     res.write(`data: [DONE]\n\n`);
                     res.end();
                     console.log("<-- Stream finished successfully.");
                 } else {
                      console.log("<-- Stream finished, but connection was already ended.");
                 }

            } catch (streamError) {
                // Catches errors from makeFalRequestWithRetry OR the stream iteration itself
                console.error('Error during stream request processing:', streamError.message || streamError);
                 try {
                     if (!res.headersSent) {
                         // Error likely occurred in makeFalRequestWithRetry before stream started
                         res.status(502).json({ // 502 Bad Gateway might be appropriate
                            error: 'Failed to initiate Fal stream',
                            details: streamError.message || 'Underlying Fal request failed or timed out.'
                         });
                         console.log("<-- Stream initiation failed response sent.");
                     } else if (!res.writableEnded) {
                         // Stream started but failed during processing
                         const errorDetails = (streamError instanceof Error) ? streamError.message : JSON.stringify(streamError);
                         // Send error details in the stream if possible
                         res.write(`data: ${JSON.stringify({ error: { message: "Stream processing error after initiation", type: "proxy_error", details: errorDetails } })}\n\n`);
                         res.write(`data: [DONE]\n\n`); // Still send DONE after error for client handling
                         res.end();
                         console.log("<-- Stream error sent, stream ended.");
                     } else {
                        console.log("<-- Stream error occurred, but connection already ended.");
                     }
                 } catch (finalError) {
                    console.error('Error sending stream error message to client:', finalError);
                    // Ensure response is ended if possible
                    if (!res.writableEnded) { res.end(); }
                 }
            }

        } else {
            // --- Non-Stream ---
            try {
                // Get the result using the retry logic
                const result = await makeFalRequestWithRetry(falInput, false);
                // console.log("Received non-stream result via retry function:", JSON.stringify(result, null, 2)); // Verbose

                // Construct OpenAI compatible response
                const openAIResponse = {
                    id: `chatcmpl-${result.requestId || Date.now()}`,
                    object: "chat.completion",
                    created: Math.floor(Date.now() / 1000),
                    model: model, // Echo back requested model
                    choices: [{
                        index: 0,
                        message: {
                            role: "assistant",
                            content: result.output || "" // Ensure content is string
                         },
                        finish_reason: "stop" // Assume stop for non-stream success
                    }],
                    usage: { // Provide null usage as Fal doesn't return it
                        prompt_tokens: null,
                        completion_tokens: null,
                        total_tokens: null
                     },
                    system_fingerprint: null, // Fal doesn't provide this
                     ...(result.reasoning && { fal_reasoning: result.reasoning }), // Include Fal specific reasoning if present
                };

                res.json(openAIResponse);
                console.log("<-- Non-stream response sent successfully.");

            } catch (error) {
                 // Catches errors from makeFalRequestWithRetry (e.g., all keys failed or non-key error)
                console.error('Error during non-stream request processing:', error.message || error);
                if (!res.headersSent) {
                    const errorMessage = (error instanceof Error) ? error.message : JSON.stringify(error);
                    // Check if it was the "all keys failed" error
                    const finalMessage = errorMessage.includes("No valid Fal AI keys available") || errorMessage.includes("Request failed after trying")
                        ? `Fal request failed after trying all available keys: ${errorMessage}`
                        : `Internal Server Error processing Fal request: ${errorMessage}`;
                    // Use 502 Bad Gateway if it's likely an upstream (Fal) failure
                    res.status(502).json({ error: 'Fal Request Failed', details: finalMessage });
                    console.log("<-- Non-stream error response sent.");
                } else {
                    // Should be rare for non-stream, but handle just in case
                    console.error("Headers already sent for non-stream error? This is unexpected.");
                    if (!res.writableEnded) { res.end(); }
                }
            }
        }

    } catch (error) {
        // Catch errors from parameter validation or prompt conversion *before* calling Fal
        console.error('Unhandled error before initiating Fal request:', error.message || error);
        if (!res.headersSent) {
            const errorMessage = (error instanceof Error) ? error.message : JSON.stringify(error);
            res.status(500).json({ error: 'Internal Server Error in Proxy Setup', details: errorMessage });
            console.log("<-- Proxy setup error response sent.");
        } else {
             console.error("Headers already sent when catching setup error. Ending response.");
             if (!res.writableEnded) { res.end(); }
        }
    }
});

// 启动服务器 (Updated startup message)
app.listen(PORT, () => {
    console.log(`=====================================================================`);
    console.log(` Fal OpenAI Proxy Server (Multi-Key Rotation & Failover)`);
    console.log(`---------------------------------------------------------------------`);
    console.log(` Listening on port : ${PORT}`);
    console.log(` Reading Fal Keys from : FAL_KEY environment variable (comma-separated)`);
    console.log(` Loaded Keys Count   : ${falKeys.length}`);
    console.log(` API Key Auth        : ${API_KEY ? 'Enabled (using API_KEY env var)' : 'Disabled'}`);
    console.log(` Input Limits        : System Prompt=${SYSTEM_PROMPT_LIMIT}, Prompt=${PROMPT_LIMIT}`);
    console.log(` Concurrency Warning : Global Fal client reconfigured per request.`);
    console.log(`---------------------------------------------------------------------`);
    console.log(` Endpoints:`);
    console.log(`   POST http://localhost:${PORT}/v1/chat/completions`);
    console.log(`   GET  http://localhost:${PORT}/v1/models`);
    console.log(`=====================================================================`);
});

// 根路径响应 (Updated message)
app.get('/', (req, res) => {
    res.send(`Fal OpenAI Proxy (Multi-Key Rotation from FAL_KEY) is running. Loaded ${falKeys.length} key(s).`);
});
