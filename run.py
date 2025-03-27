from app import create_app

# Create the application instance
app = create_app()

# Push an application context to ensure 
# resources like the database are properly initialized
app_context = app.app_context()
app_context.push()

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)