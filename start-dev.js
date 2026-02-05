const { spawn } = require('child_process');
const path = require('path');

console.log('🚀 Starting Physical AI & Humanoid Robotics Development Environment...\n');

// Start API Server
console.log('📡 Starting Chat API Server...');
const apiServer = spawn('node', ['api-server.js'], {
  stdio: 'inherit',
  cwd: __dirname
});

apiServer.on('error', (error) => {
  console.error('❌ Failed to start API server:', error);
  process.exit(1);
});

// Wait a moment for API server to start
setTimeout(() => {
  console.log('\n🌐 Starting Docusaurus Development Server...');
  
  // Start Docusaurus
  const docusaurus = spawn('npm', ['run', 'start'], {
    stdio: 'inherit',
    cwd: __dirname,
    shell: true
  });

  docusaurus.on('error', (error) => {
    console.error('❌ Failed to start Docusaurus:', error);
    apiServer.kill();
    process.exit(1);
  });

  // Handle shutdown
  process.on('SIGINT', () => {
    console.log('\n🛑 Shutting down servers...');
    apiServer.kill();
    docusaurus.kill();
    process.exit(0);
  });

  process.on('SIGTERM', () => {
    console.log('\n🛑 Shutting down servers...');
    apiServer.kill();
    docusaurus.kill();
    process.exit(0);
  });
}, 2000);

console.log('\n✅ Development servers are starting...');
console.log('📚 Chat API will be available at: http://localhost:8000');
console.log('🌐 Documentation site will be available at: http://localhost:3000');
console.log('\n💡 Press Ctrl+C to stop both servers\n');
