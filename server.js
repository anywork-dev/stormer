const express = require('express');
const path = require('path');
const session = require('express-session');
const { authenticateToken, requireAuth, generateToken } = require('./middleware/auth');
const { initializeDemoUser, findUserByEmail, validatePassword, createUser } = require('./utils/users');
const { decode, encode } = require('./toon-lib.js');
const dotenv = require('dotenv');
const crypto = require('crypto');
const { count } = require('console');

dotenv.config()

// API Configuration and Functions
const GEMINI_API_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent';

/**
 * Call Gemini API with system prompt, user message, and optional conversation history
 * @param {string} apiKey - Gemini API key
 * @param {string} systemPrompt - System instruction prompt
 * @param {string} userMessage - User message
 * @param {Array} history - Conversation history
 * @param {boolean} enableGrounding - Enable Google Search grounding (default: false)
 */
async function callGeminiAPI(apiKey, systemPrompt, userMessage, history = [], enableGrounding = false) {
    // Combine system prompt and user message for single-turn conversation
    const combinedPrompt = `${systemPrompt}\n\n${userMessage}`;
    
    const contents = [];
    
    // Add conversation history with proper roles
    for (const msg of history) {
        contents.push({
            role: msg.role === 'user' ? 'user' : 'model',
            parts: [{ text: msg.content }]
        });
    }
    
    // Add current message as user role
    contents.push({
        role: 'user',
        parts: [{ text: combinedPrompt }]
    });

    // Build request body
    const requestBody = {
        contents
    };

    // Add Google Search grounding tool if enabled
    if (enableGrounding) {
        requestBody.tools = [
            {
                google_search: {}
            }
        ];
    }

    const response = await fetch(GEMINI_API_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'x-goog-api-key': apiKey
        },
        body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error?.message || `API request failed with status ${response.status}`);
    }

    const data = await response.json();
    
    if (!data.candidates || !data.candidates[0] || !data.candidates[0].content) {
        throw new Error('Invalid response from Gemini API');
    }

    return data.candidates[0].content.parts[0].text;
}

function shuffleArray(array = [], count = 3, topic = '') {
    const items = Array.isArray(array) ? array.slice() : [];
    count = Number.isFinite(+count) ? Math.max(0, Math.floor(+count)) : 3;
    if (count === 0 || items.length === 0) return [];

    const rawTopic = String(topic || '');
    const useAnd = rawTopic.includes('+');
    const keywords = rawTopic
        .split(/\+|\s*,\s*|\s+/)
        .map(k => k.trim())
        .filter(Boolean);

    const escapeRegex = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

    // Choose pool: prefer matched items if any keywords provided and matches exist
    let pool = items;


    if (keywords.length > 0) {
        if (useAnd) {
            const pattern = keywords.map(k => `(?=.*${escapeRegex(k)})`).join('');
            const re = new RegExp(pattern, 'i');
            const matches = items.filter(s => re.test(`${s.title || ''} ${s.content || ''}`));
            if (matches.length >= count) pool = matches;
        } else {
            const pattern = keywords.map(escapeRegex).join('|');
            const re = new RegExp(pattern, 'i');
            const matches = items.filter(s => re.test(`${s.title || ''} ${s.content || ''}`));
            if (matches.length >= count) pool = matches;
        }
    }

    // Fisher-Yates shuffle
    for (let i = pool.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [pool[i], pool[j]] = [pool[j], pool[i]];
    }

    return pool.slice(0, Math.min(count, pool.length));
}


/**
    * Fetch sources from Hacker News (via hn.algolia.com) based on a topic.
    * Returns items with title, url and any HN-provided content (story/comment text).
    * Caches results globally for 24 hours (in-memory) — not keyed by topic.
    */
    const HN_CACHE_TTL_MS = 24 * 60 * 60 * 1000;
    const hnCache = new Map();
    const HN_CACHE_KEY = '__hn_all__';

    async function fetchHnSources(topic, count = 3) {
        // Use a single global cache key so all topics share the same cached result
        const key = HN_CACHE_KEY;

        // Return cached results if still valid
        const cached = hnCache.get(key);
        if (cached && cached.expires > Date.now()) {
            // Return a slice so callers asking for a smaller count still work
            return shuffleArray(cached.sources, count, topic);
        }

        const endpointsMap = {
            top: 'topstories',
            new: 'newstories',
            best: 'beststories',
            ask: 'askstories',
            show: 'showstories',
            job: 'jobstories',
        };
        // still allow selecting an endpoint based on topic, but caching is global
        const listEndpoint = endpointsMap[(topic ?? '').toLowerCase()] ?? 'topstories';
        const base = 'https://hacker-news.firebaseio.com/v0';

        // fetch list of story IDs
        const idsResp = await fetch(`${base}/${listEndpoint}.json`);
        if (!idsResp.ok) {
            throw new Error(`Failed to fetch story IDs from HN Firebase API: ${idsResp.status} ${idsResp.statusText}`);
        }
        const ids = await idsResp.json();
        if (!Array.isArray(ids)) {
            throw new Error('Unexpected IDs response from HN Firebase API');
        }

        // limit to up to 500 IDs (HN top/new lists can be up to ~500)
        const idsSlice = ids.slice(0, 500);

        // fetch item details in parallel (gracefully ignore failures)
        const itemPromises = idsSlice.map((id) =>
            fetch(`${base}/item/${id}.json`)
                .then(r => r.ok ? r.json() : null)
                .catch(() => null)
        );
        const items = (await Promise.all(itemPromises)).filter(Boolean);

        // normalize to an Algolia-like shape so the rest of the function can reuse existing mapping
        const hits = items.map((it) => ({
            title: it.title ?? null,
            url: it.url ?? null,
            text: it.text ?? null,
            story_title: it.title ?? null,
            story_url: it.url ?? null,
            story_text: it.text ?? null,
            comment_text: it.text ?? null,
            author: it.by ?? null,
            created_at: it.time ? new Date(it.time * 1000).toISOString() : null,
            objectID: it.id ?? null,
        }));

        // build a response-like object expected by the existing code
        const response = {
            ok: true,
            status: 200,
            statusText: 'OK',
            json: async () => ({ hits }),
        };
        if (!response.ok) {
            throw new Error(`Failed to fetch from Hacker News API: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        if (!Array.isArray(data.hits)) {
            throw new Error('Unexpected response shape from Hacker News API');
        }

        const sources = data.hits.map((hit) => {
            const title = hit.title ?? hit.story_title ?? null;
            const url = hit.url ?? hit.story_url ?? `https://news.ycombinator.com/item?id=${hit.objectID}`;
            // Algolia returns the item text in different fields depending on type
            const content = hit.text ?? hit.story_text ?? hit.comment_text ?? null;
            return {
                title,
                url,
                content,
                author: hit.author ?? null,
                created_at: hit.created_at ?? null,
            };
        }).filter((s) => s.title && s.url);

        // Deduplicate by URL while preserving order
        const uniqueByUrl = Array.from(new Map(sources.map((s   ) => [s.url, s])).values());

        // Cache the full result for TTL and return requested slice (global cache)
        hnCache.set(key, { expires: Date.now() + HN_CACHE_TTL_MS, sources: uniqueByUrl });

            
        return shuffleArray(uniqueByUrl, count, topic);
    }


async function generateNewIdea(topic = 'innovative startup ideas') {
        try {
            // Fetch Hacker News sources first
            const hnSources = await fetchHnSources(topic, 5);
            const hackerNewsContext = hnSources
                .map(s => `Title: ${s.title}\nURL: ${s.url}${s.content ? `\nContent: ${s.content}` : ''}`)
                .join('\n\n');

            // Build the system prompt
            const systemPrompt = `You are an expert venture analyst and idea synthesizer. Your task: search the public internet for real people's problems related to the topic "${topic}" (forums, news, social discussions, Q&A, product reviews, Hacker News, Reddit threads, etc.), synthesize findings, and generate a single, novel business idea based on that evidence.

Use Google Search grounding and the provided Hacker News context to find concrete pain points people report. Prioritize recent, public, and relevant posts or articles. Cite at least 3 distinct public sources (title, url, short summary of the evidence).

Follow these rules strictly:
1. Your response MUST be in valid TOON format with 2-space indentation.
2. Use a single indented, pipe-delimited data row (NOT line-by-line key:value pairs).
3. Include these 9 fields in order: id|name|jargon|problem|market|how|impact|valuation|sources
4. The sources field must be a JSON array string like: [{"title":"...","url":"...","summary":"..."}]
5. Use this exact header: ideas[1|]{id|name|jargon|problem|market|how|impact|valuation|sources}:

CRITICAL FORMAT (2-space indent, pipe-delimited row):

ideas[1|]{id|name|jargon|problem|market|how|impact|valuation|sources}:
  ${crypto.randomUUID()}|StartupName|One-line jargon/tagline|Concise problem statement summarizing people's reported pain|Target market description|How the product/service works (one short sentence)|Impact and benefits (one sentence)|$50M-$100M|[{"title":"Source Title","url":"https://example.com","summary":"One-line summary of evidence"}]

IMPORTANT:
- Single data row only, pipe (|) as delimiter, 2-space indentation before the row.
- Keep each field concise (one sentence or short phrase).
- Include at least 3 sources discovered on the public web; if insufficient web evidence exists, explicitly note that and rely on the supplied Hacker News context.
- Do NOT invent source URLs — only include real public URLs or fallback to Hacker News context entries.
- Begin your response with the exact TOON header followed by ONE indented pipe-delimited data row.`;

            // User message with the actual context
            // const userMessage = `Here are the Hacker News sources about "${topic}":\n\n${hackerNewsContext}\n\nGenerate ONE business idea in TOON format. Remember: header line, then ONE indented comma-separated data row.`;
            const userMessage = `Here are the Hacker News sources about "${topic}":\n\n${hackerNewsContext}\n\nUsing the Hacker News context above plus web search grounding, synthesize evidence of real people's pain points and generate ONE novel, actionable business idea. Follow these requirements:\n- Respond ONLY in TOON format with the exact header line and a SINGLE 2-space-indented pipe-delimited data row as specified in the system prompt.\n- Include at least 3 distinct public sources in the sources field (a JSON array string). Do NOT invent URLs; if web evidence is insufficient, explicitly state that and rely on the Hacker News context.\n- Keep each field concise (one sentence or short phrase).\nProvide the single TOON-formatted idea now.`;

            console.log('System Prompt:', systemPrompt);
            console.log('User Message:', userMessage);

            // Call Gemini API with Google Search grounding enabled
            const response = await callGeminiAPI(process.env.GEMINI_API_KEY, systemPrompt, userMessage, [], true);

            console.log('Gemini API Response:', response);

            // Try to decode the TOON response
            const decoded = decode(response);

            // Handle both array and object responses
            let card;
            if (Array.isArray(decoded) && decoded.length > 0) {
                card = decoded[0];
            } else if (decoded && typeof decoded === 'object' && !Array.isArray(decoded)) {
                card = decoded.ideas[0];
            } else {
                throw new Error('Invalid card format received from AI');
            }

            // Ensure all required fields exist
            if (!card.id) {
                card.id = this.generateUUID();
            }
            if (!card.name) {
                throw new Error('Missing required field: name');
            }
            
            // Add timestamp
            card.createdAt = new Date().toISOString();

            // Ensure sources is an array
            if (typeof card.sources === 'string') {
                try {
                    card.sources = JSON.parse(card.sources);
                } catch (e) {
                    card.sources = hnSources.slice(0, 3); // Fallback to HN sources
                }
            }
            if (!Array.isArray(card.sources)) {
                card.sources = hnSources.slice(0, 3);
            }

            return card;

        } catch (error) {
            console.error('Error in generateNewIdea:', error);
            console.error('Error details:', error.message);
            throw new Error(`Failed to generate idea: ${error.message}`);
        }
    }

    
    /**
     * Save new idea from brainstorm
     */
    async function saveNewIdea() {
        const systemPrompt = `Now synthesize our entire discussion into a final, refined idea. You MUST respond with ONLY a TOON-formatted string for the new card. Use this exact header: cards[1]{id,title,summary,tags,source,sources}: 

Requirements:
- Set the source field to "From Brainstorm"
- The sources field must be an empty array []
- Include ALL required fields: id, title, summary, tags, source, sources
- Make the idea clear, actionable, and well-structured`;

        const message = 'Please create the final synthesized idea card based on our discussion. Remember to use TOON format with all required fields.';

        const response = await callGeminiAPI(this.state.apiKey, systemPrompt, message, this.state.conversationHistory);

        try {
            const decoded = decode(response);

            // Handle both array and object responses
            let card;
            if (Array.isArray(decoded) && decoded.length > 0) {
                card = decoded[0];
            } else if (decoded && typeof decoded === 'object' && !Array.isArray(decoded)) {
                card = decoded;
            } else {
                throw new Error('Invalid card format received from AI');
            }

            // Add timestamp and ensure ID exists
            if (!card.id) {
                card.id = this.generateUUID();
            }
            card.createdAt = new Date().toISOString();
            card.source = 'From Brainstorm';

            return card;
        } catch (error) {
            console.error('Failed to decode TOON response:', response);
            throw new Error(`Failed to parse AI response: ${error.message}`);
        }
    }

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Session configuration
app.use(session({
    secret: process.env.SESSION_SECRET || 'your-session-secret-change-in-production',
    resave: false,
    saveUninitialized: false,
    cookie: {
        secure: process.env.NODE_ENV === 'production', // HTTPS only in production
        maxAge: 24 * 60 * 60 * 1000 // 24 hours
    }
}));

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, 'public')));

// Authentication routes
app.get('/login', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'login.html'));
});

app.post('/api/login', async (req, res) => {
    try {
        const { email, password, remember } = req.body;

        if (!email || !password) {
            return res.status(400).json({
                message: 'Email and password are required'
            });
        }

        // Find user
        const user = findUserByEmail(email);
        if (!user) {
            return res.status(401).json({
                message: 'Invalid email or password'
            });
        }

        // Validate password
        const isValidPassword = await validatePassword(user, password);
        if (!isValidPassword) {
            return res.status(401).json({
                message: 'Invalid email or password'
            });
        }

        // Create session
        req.session.user = {
            id: user.id,
            email: user.email,
            name: user.name
        };

        // Generate JWT token
        const token = generateToken({
            id: user.id,
            email: user.email,
            name: user.name
        });

        // Set cookie if remember me is checked
        if (remember) {
            res.cookie('auth_token', token, {
                httpOnly: true,
                maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
                secure: process.env.NODE_ENV === 'production'
            });
        }

        res.json({
            message: 'Login successful',
            user: {
                id: user.id,
                email: user.email,
                name: user.name
            },
            token,
            redirect: '/welcome'
        });

    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({
            message: 'Internal server error'
        });
    }
});

app.post('/api/logout', (req, res) => {
    req.session.destroy((err) => {
        if (err) {
            return res.status(500).json({ message: 'Error logging out' });
        }
        res.clearCookie('auth_token');
        res.json({ message: 'Logged out successfully' });
    });
});

app.post('/api/register', async (req, res) => {
    try {
        const { email, password, name } = req.body;

        if (!email || !password) {
            return res.status(400).json({
                message: 'Email and password are required'
            });
        }

        if (password.length < 6) {
            return res.status(400).json({
                message: 'Password must be at least 6 characters long'
            });
        }

        // Create new user
        const user = await createUser({ email, password, name });

        // Create session
        req.session.user = {
            id: user.id,
            email: user.email,
            name: user.name
        };

        // Generate JWT token
        const token = generateToken({
            id: user.id,
            email: user.email,
            name: user.name
        });

        res.status(201).json({
            message: 'Registration successful',
            user: {
                id: user.id,
                email: user.email,
                name: user.name
            },
            token,
            redirect: '/welcome'
        });

    } catch (error) {
        console.error('Registration error:', error);
        if (error.message === 'User already exists with this email') {
            return res.status(409).json({ message: error.message });
        }
        res.status(500).json({
            message: 'Internal server error'
        });
    }
});

// Get current user info
app.get('/api/me', authenticateToken, (req, res) => {
    res.json({
        user: req.user
    });
});

// Serve all HTML files from public directory
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/discovery', requireAuth, (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'discovery.html'));
});

app.get('/collection', requireAuth, (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'collection.html'));
});

app.get('/workbench', requireAuth, (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'workbench.html'));
});

app.get('/profile', requireAuth, (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'profile.html'));
});

app.get('/welcome', requireAuth, (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'welcome.html'));
});

app.get('/exploring', requireAuth, (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'exploring.html'));
});

// API route for AI idea generation
app.get('/api/generate-idea', requireAuth, async (req, res) => {
    try {
        const topic = (req.query.topic || req.body?.topic || 'innovative startup ideas').toString();
        console.log(`Generating idea for topic: "${topic}" (user: ${req.user?.email || 'unknown'})`);

        const ideaCard = await generateNewIdea(topic);

        // Attach metadata about requesting user
        ideaCard.requestedBy = {
            id: req.user?.id,
            email: req.user?.email,
            name: req.user?.name
        };

        res.json({ idea: ideaCard });
    } catch (error) {
        console.error('Error generating idea:', error);
        res.status(500).json({ error: 'Failed to generate idea' });
    }
});

// Initialize demo user and start server
const startServer = async () => {
    await initializeDemoUser();

    app.listen(PORT, () => {
        console.log(`Inspedia server running on http://localhost:${PORT}`);
        console.log(`Available routes:`);
        console.log(`  / - Main page`);
        console.log(`  /login - Login page`);
        console.log(`  /welcome - Welcome page (requires auth)`);
        console.log(`  /discovery - Idea discovery (requires auth)`);
        console.log(`  /collection - Saved ideas (requires auth)`);
        console.log(`  /workbench - Idea workbench (requires auth)`);
        console.log(`  /profile - User profile (requires auth)`);
        console.log(`  /api/generate-idea - AI idea generation API (requires auth)`);
        console.log(`  /api/login - User login`);
        console.log(`  /api/register - User registration`);
        console.log(`  /api/logout - User logout`);
        console.log(`  /api/me - Get current user info`);
        console.log(`\nDemo credentials:`);
        console.log(`  Email: demo@inspedia.com`);
        console.log(`  Password: demo123`);
    });
};

// Start server
startServer();