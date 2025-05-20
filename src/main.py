import os
import argparse
from dotenv import load_dotenv
from agent.data_agent import DataAgent

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key.")
        return
    
    # Initialize the data agent
    agent = DataAgent()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DataSen - Intelligent Data Analysis Assistant")
    parser.add_argument("--data", help="Path to the data file (CSV format)")
    args = parser.parse_args()
    
    print("Welcome to DataSen - Your Intelligent Data Analysis Assistant!")
    print("Type 'exit' to quit, 'help' for available commands.")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYour question: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("- exit: Exit the program")
                print("- help: Show this help message")
                print("\nExample questions:")
                print("- What are the main trends in this dataset?")
                print("- Show me a summary of the numerical columns")
                print("- Find correlations between variables")
                print("- What are the outliers in this dataset?")
                continue
            
            # Process the query
            if args.data:
                response = agent.analyze(user_input, args.data)
            else:
                response = agent.analyze(user_input)
            
            print("\nAssistant:", response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or type 'help' for available commands.")

if __name__ == "__main__":
    main() 