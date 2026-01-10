import json
import aisuite as ai
import display_functions
import email_tools
import utils

from dotenv import load_dotenv


load_dotenv()
client = ai.Client()

## ------------------------------------------------------ ##
new_email_id = utils.test_send_email()

# uncomment the line 'utils.test_*' you want to try
_ = utils.test_get_email(new_email_id['id'])
#_ = utils.test_list_emails()
#_ = utils.test_filter_emails(recipient="test@example.com")
#_ = utils.test_search_emails("lunch")
#_ = utils.test_unread_emails()
#_ = utils.test_mark_read(new_email_id['id'])
#_ = utils.test_mark_unread(new_email_id['id'])
#_ = utils.test_delete_email(new_email_id['id'])
#_ = utils.reset_database()

## ------------------------------------------------------ ##
new_email = email_tools.send_email("test@example.com", "Lunch plans", "Shall we meet at noon?")
content_ = email_tools.get_email(new_email['id'])

# Uncomment the ones you want to try:
content_ = email_tools.list_all_emails()
#content_ = email_tools.list_unread_emails()
#content_ = email_tools.search_emails("lunch")
#content_ = email_tools.filter_emails(recipient="test@example.com")
#content_ = email_tools.mark_email_as_read(new_email['id'])
#content_ = email_tools.mark_email_as_unread(new_email['id'])
#content_ = email_tools.search_unread_from_sender("test@example.com")
#content_ = email_tools.delete_email(new_email['id'])

utils.print_html(content = json.dumps(content_, indent = 2), title = "Testing the email_tools")

## ------------------------------------------------------ ##
def build_prompt(request_: str) -> str:
    return f"""
            - You are an AI assistant specialized in managing emails.
            - You can perform various actions such as listing, searching, filtering, and \
                manipulating emails.
            - Use the provided tools to interact with the email system.
            - Never ask the user for confirmation before performing an action.
            - If needed, my email address is "you@email.com" so you can use it to send emails or \
                perform actions related to my account.

            {request_.strip()}
            """

## ------------------------------------------------------ ##
example_prompt = build_prompt("Delete the Happy Hour email")
utils.print_html(content = example_prompt, title = "Example example_prompt")

## ------------------------------------------------------ ##
utils.reset_database()

## ------------------------------------------------------ ##
prompt_ = build_prompt("Check for unread emails from boss@email.com, mark them as read, and send " \
                        "a polite follow-up.")

response = client.chat.completions.create(model = "openai:gpt-4.1",
                                        messages = [{"role" : "user", "content" : (prompt_)}],
                                        tools = [email_tools.search_unread_from_sender,
                                                email_tools.list_unread_emails,
                                                email_tools.search_emails,
                                                email_tools.get_email,
                                                email_tools.mark_email_as_read,
                                                email_tools.send_email],
                                        max_turns = 5, )

display_functions.pretty_print_chat_completion(response)

## ------------------------------------------------------ ##
prompt_ = build_prompt("Delete alice@work.com email")

response = client.chat.completions.create(model = "openai:o4-mini",
                                        messages = [{"role" : "user", "content" : (prompt_)}],
                                        tools = [email_tools.search_unread_from_sender,
                                                email_tools.list_unread_emails,
                                                email_tools.search_emails,
                                                email_tools.get_email,
                                                email_tools.mark_email_as_read,
                                                email_tools.send_email],
                                        max_turns = 5)

display_functions.pretty_print_chat_completion(response)

## ------------------------------------------------------ ##
prompt_ = build_prompt("Delete alice@work.com email")

response = client.chat.completions.create(model = "openai:o4-mini",
                                        messages = [{"role" : "user", "content" : (prompt_)}],
                                        tools = [email_tools.search_unread_from_sender,
                                                email_tools.list_unread_emails,
                                                email_tools.search_emails,
                                                email_tools.get_email,
                                                email_tools.mark_email_as_read,
                                                email_tools.send_email,
                                                email_tools.delete_email],
                                        max_turns = 5)

display_functions.pretty_print_chat_completion(response)

## ------------------------------------------------------ ##
prompt_ = build_prompt("Delete the happy hour email")

response = client.chat.completions.create(model = "openai:o4-mini",
                                        messages = [{"role" : "user", "content" : (prompt_)}],
                                        tools = [email_tools.search_unread_from_sender,
                                                email_tools.list_unread_emails,
                                                email_tools.search_emails,
                                                email_tools.get_email,
                                                email_tools.mark_email_as_read,
                                                email_tools.send_email,
                                                email_tools.delete_email],
                                        max_turns = 5)

display_functions.pretty_print_chat_completion(response)
