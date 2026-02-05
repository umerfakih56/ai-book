# Railway Deployment Guide

## Prerequisites
- GitHub account
- Railway account (free tier available)
- Backend code pushed to GitHub

## Step 1: Push to GitHub

1. Create a new repository on GitHub
2. Push the rag-backend folder to GitHub:

```bash
cd rag-backend
git init
git add .
git commit -m "Initial commit: RAG ChatBot Backend"
git branch -M main
git remote add origin https://github.com/yourusername/rag-backend.git
git push -u origin main
```

## Step 2: Deploy to Railway

1. **Sign up/login to Railway**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Create New Project**
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your rag-backend repository
   - Railway will automatically detect the Dockerfile

3. **Configure Deployment**
   - Railway will build and deploy automatically
   - Wait for deployment to complete (2-3 minutes)

## Step 3: Add Environment Variables

In your Railway project dashboard:

1. Go to "Variables" tab
2. Add these environment variables:

```
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
DATABASE_URL=postgresql://username:password@localhost:5432/database_name
```

**Important:**
- For Gemini API: Get key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- For Qdrant: Use [Qdrant Cloud](https://cloud.qdrant.io/) or self-hosted
- For PostgreSQL: Use Railway's built-in PostgreSQL or external

## Step 4: Get Public URL

1. After deployment, Railway will provide a public URL
2. URL format: `https://your-app-name.up.railway.app`
3. Test the health endpoint: `https://your-app-name.up.railway.app/health`

## Step 5: Update Vercel Environment Variables

1. Go to your Vercel dashboard
2. Select your Docusaurus project
3. Go to "Settings" → "Environment Variables"
4. Add/update:
   ```
   REACT_APP_API_URL=https://your-app-name.up.railway.app
   ```
5. Redeploy Vercel site

## Alternative: Render Deployment

If you prefer Render, create `render.yaml`:

```yaml
services:
  - type: web
    name: rag-backend
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /health
    envVars:
      - key: GEMINI_API_KEY
        sync: false
      - key: QDRANT_URL
        sync: false
      - key: QDRANT_API_KEY
        sync: false
      - key: DATABASE_URL
        sync: false
```

## Testing the Deployment

1. **Health Check**: Visit `/health` endpoint
2. **API Test**: Test chat endpoint:
   ```bash
   curl -X POST https://your-app-name.up.railway.app/api/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "What is ROS 2?", "session_id": "test"}'
   ```

## Troubleshooting

### Common Issues:

1. **Build Fails**:
   - Check Dockerfile syntax
   - Verify requirements.txt has all dependencies

2. **Runtime Errors**:
   - Check environment variables in Railway dashboard
   - View logs in Railway console

3. **Database Connection**:
   - Verify DATABASE_URL format
   - Check if PostgreSQL is running

4. **CORS Issues**:
   - Ensure your frontend URL is in CORS allow list
   - Check main.py CORS configuration

### Railway Logs:
- Go to your project → "Logs" tab
- View real-time logs for debugging

## Cost Considerations

**Railway Free Tier**:
- $5 credit/month (enough for development)
- 500 hours of runtime
- 100GB bandwidth

**Production Scaling**:
- Monitor usage in Railway dashboard
- Upgrade plans as needed

## Next Steps

1. Set up monitoring and alerting
2. Configure custom domain (optional)
3. Set up CI/CD for automatic deployments
4. Add database migrations
5. Implement rate limiting for production
