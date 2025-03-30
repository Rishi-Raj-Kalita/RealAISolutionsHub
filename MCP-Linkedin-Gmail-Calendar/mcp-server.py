from mcp.server.fastmcp import FastMCP
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
import re
from email.mime.text import MIMEText
from datetime import datetime
from linkedin_api import Linkedin
from dotenv import load_dotenv
import base64
import logging
import traceback
import logging
import traceback
import pytz
import os

load_dotenv()

mcp = FastMCP("MCP-Linkedin-Gmail-Calendar")

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly', "openid",
    "https://mail.google.com/", "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/userinfo.email"
]


def get_creds():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(
            "/Users/rishirajkalita/Desktop/RealAISolutionsHub/MCP-Linkedin-Gmail-Calendar/credentials/token.json"
    ):
        creds = Credentials.from_authorized_user_file(
            "/Users/rishirajkalita/Desktop/RealAISolutionsHub/MCP-Linkedin-Gmail-Calendar/credentials/token.json",
            SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "/Users/rishirajkalita/Desktop/RealAISolutionsHub/MCP-Linkedin-Gmail-Calendar/credentials/credentials.json",
                SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(
                "/Users/rishirajkalita/Desktop/RealAISolutionsHub/MCP-Linkedin-Gmail-Calendar/credentials/token.json",
                "w") as token:
            token.write(creds.to_json())

    return creds


def _extract_body(payload) -> str | None:
    """
        Extract the email body from the payload.
        Handles both multipart and single part messages, including nested multiparts.
        """
    try:
        # For single part text/plain messages
        if payload.get('mimeType') == 'text/plain':
            data = payload.get('body', {}).get('data')
            if data:
                return base64.urlsafe_b64decode(data).decode('utf-8')

        # For multipart messages (both alternative and related)
        if payload.get('mimeType', '').startswith('multipart/'):
            parts = payload.get('parts', [])

            # First try to find a direct text/plain part
            for part in parts:
                if part.get('mimeType') == 'text/plain':
                    data = part.get('body', {}).get('data')
                    if data:
                        return base64.urlsafe_b64decode(data).decode('utf-8')

            # If no direct text/plain, recursively check nested multipart structures
            for part in parts:
                if part.get('mimeType', '').startswith('multipart/'):
                    nested_body = _extract_body(part)
                    if nested_body:
                        return nested_body

            # If still no body found, try the first part as fallback
            if parts and 'body' in parts[0] and 'data' in parts[0]['body']:
                data = parts[0]['body']['data']
                return base64.urlsafe_b64decode(data).decode('utf-8')

        return None

    except Exception as e:
        logging.error(f"Error:  {str(e)}")
        logging.error(traceback.format_exc())
        return None


def _parse_message(txt, parse_body=False) -> dict | None:
    """
        Parse a Gmail message into a structured format.
        
        Args:
            txt (dict): Raw message from Gmail API
            parse_body (bool): Whether to parse and include the message body (default: False)
        
        Returns:
            dict: Parsed message containing comprehensive metadata
            None: If parsing fails
        """
    try:
        message_id = txt.get('id')
        thread_id = txt.get('threadId')
        payload = txt.get('payload', {})
        headers = payload.get('headers', [])

        metadata = {
            'id': message_id,
            'threadId': thread_id,
            'historyId': txt.get('historyId'),
            'internalDate': txt.get('internalDate'),
            'sizeEstimate': txt.get('sizeEstimate'),
            'labelIds': txt.get('labelIds', []),
            'snippet': txt.get('snippet'),
        }

        for header in headers:
            name = header.get('name', '').lower()
            value = header.get('value', '')

            if name == 'subject':
                metadata['subject'] = value
            elif name == 'from':
                metadata['from'] = value
            elif name == 'to':
                metadata['to'] = value
            elif name == 'date':
                metadata['date'] = value
            elif name == 'cc':
                metadata['cc'] = value
            elif name == 'bcc':
                metadata['bcc'] = value
            elif name == 'message-id':
                metadata['message_id'] = value
            elif name == 'in-reply-to':
                metadata['in_reply_to'] = value
            elif name == 'references':
                metadata['references'] = value
            elif name == 'delivered-to':
                metadata['delivered_to'] = value

        if parse_body:
            body = _extract_body(payload)
            if body:
                metadata['body'] = body

            metadata['mimeType'] = payload.get('mimeType')

        return metadata

    except Exception as e:
        logging.error(f"Error:  {str(e)}")
        logging.error(traceback.format_exc())
        return None


@mcp.tool()
def query_emails(query: str):
    """
        Query emails from Gmail based on a search query.
        
        Args:
            query (str, optional): Gmail search query (e.g., 'is:unread', 'from:example@gmail.com')
                                If None, returns all emails
            max_results (int): Maximum number of emails to retrieve (1-500, default: 100)
        
        Returns:
            list: List of parsed email messages, newest first
        """
    try:
        # Call the Gmail API
        creds = get_creds()
        service = build("gmail", "v1", credentials=creds)
        result = service.users().messages().list(userId='me',
                                                 maxResults=5,
                                                 q=query).execute()
        messages = result.get('messages', [])
        parsed = []
        for msg in messages:
            txt = service.users().messages().get(userId='me',
                                                 id=msg['id']).execute()
        parsed_messaged = _parse_message(txt=txt, parse_body=False)
        if parsed_messaged:
            parsed.append(parsed_messaged)

        return parsed
    except HttpError as error:
        print(f"An error occurred: {error}")


@mcp.tool()
def create_draft(to: str,
                 subject: str,
                 body: str,
                 cc: list[str] | None = None) -> dict | None:
    """
        Create a draft email message.
        
        Args:
            to (str): Email address of the recipient
            subject (str): Subject line of the email
            body (str): Body content of the email
            cc (list[str], optional): List of email addresses to CC
            
        Returns:
            dict: Draft message data including the draft ID if successful
            None: If creation fails
        """
    try:
        # Create message body
        creds = get_creds()
        service = build("gmail", "v1", credentials=creds)
        message = {
            'to': to,
            'subject': subject,
            'text': body,
        }
        if cc:
            message['cc'] = ','.join(cc)

        # Create the message in MIME format
        mime_message = MIMEText(body)
        mime_message['to'] = to
        mime_message['subject'] = subject
        if cc:
            mime_message['cc'] = ','.join(cc)

        # Encode the message
        raw_message = base64.urlsafe_b64encode(
            mime_message.as_bytes()).decode('utf-8')

        # Create the draft
        draft = service.users().drafts().create(userId='me',
                                                body={
                                                    'message': {
                                                        'raw': raw_message
                                                    }
                                                }).execute()

        return draft

    except Exception as e:
        logging.error(f"Error:  {str(e)}")
        logging.error(traceback.format_exc())
        return None


@mcp.tool()
def list_calendars() -> list:
    """
        Lists all calendars accessible by the user.
        
        Returns:
            list: List of calendar objects with their metadata
        """
    try:
        creds = get_creds()
        service = build("calendar", "v3", credentials=creds)
        calendar_list = service.calendarList().list().execute()

        calendars = []

        for calendar in calendar_list.get('items', []):
            if calendar.get('kind') == 'calendar#calendarListEntry':
                calendars.append({
                    'id': calendar.get('id'),
                    'summary': calendar.get('summary'),
                    'primary': calendar.get('primary', False),
                    'time_zone': calendar.get('timeZone'),
                    'etag': calendar.get('etag'),
                    'access_role': calendar.get('accessRole')
                })

        return calendars

    except Exception as e:
        logging.error(f"Error retrieving calendars: {str(e)}")
        logging.error(traceback.format_exc())
        return []


@mcp.tool()
def get_events(time_min=None,
               time_max=None,
               max_results: int = 250,
               show_deleted: bool = False,
               calendar_id: str = 'primary'):
    """
    Retrieve calendar events within a specified time range.
    
    Args:
        time_min (str, optional): Start time in RFC3339 format. Defaults to current time.
        time_max (str, optional): End time in RFC3339 format
        max_results (int): Maximum number of events to return (1-2500)
        show_deleted (bool): Whether to include deleted events
        
    Returns:
        list: List of calendar events
    """
    try:
        # If no time_min specified, use current time
        if not time_min:
            time_min = datetime.now(pytz.UTC).isoformat()

        # Ensure max_results is within limits
        max_results = min(max(1, max_results), 2500)

        # Prepare parameters
        params = {
            'calendarId': calendar_id,
            'timeMin': time_min,
            'maxResults': max_results,
            'singleEvents': True,
            'orderBy': 'startTime',
            'showDeleted': show_deleted
        }

        # Add optional time_max if specified
        if time_max:
            params['timeMax'] = time_max

        # Execute the events().list() method
        creds = get_creds()
        service = build("calendar", "v3", credentials=creds)
        events_result = service.events().list(**params).execute()

        # Extract the events
        events = events_result.get('items', [])

        # Process and return the events
        processed_events = []
        for event in events:
            processed_event = {
                'id': event.get('id'),
                'summary': event.get('summary'),
                'description': event.get('description'),
                'start': event.get('start'),
                'end': event.get('end'),
                'status': event.get('status'),
                'creator': event.get('creator'),
                'organizer': event.get('organizer'),
                'attendees': event.get('attendees'),
                'location': event.get('location'),
                'hangoutLink': event.get('hangoutLink'),
                'conferenceData': event.get('conferenceData'),
                'recurringEventId': event.get('recurringEventId')
            }
            processed_events.append(processed_event)

        return processed_events

    except Exception as e:
        logging.error(f"Error retrieving calendar events: {str(e)}")
        logging.error(traceback.format_exc())
        return []


@mcp.tool()
def create_event(summary: str,
                 start_time: str,
                 end_time: str,
                 location: str | None = None,
                 description: str | None = None,
                 attendees: list | None = None,
                 send_notifications: bool = True,
                 timezone: str | None = None,
                 calendar_id: str = 'primary') -> dict | None:
    """
    Create a new calendar event.
    
    Args:
        summary (str): Title of the event
        start_time (str): Start time in RFC3339 format
        end_time (str): End time in RFC3339 format
        location (str, optional): Location of the event
        description (str, optional): Description of the event
        attendees (list, optional): List of attendee email addresses
        send_notifications (bool): Whether to send notifications to attendees
        timezone (str, optional): Timezone for the event (e.g. 'America/New_York')
        
    Returns:
        dict: Created event data or None if creation fails
    """
    try:
        # Prepare event data
        event = {
            'summary': summary,
            'start': {
                'dateTime': start_time,
                'timeZone': timezone or 'UTC',
            },
            'end': {
                'dateTime': end_time,
                'timeZone': timezone or 'UTC',
            }
        }

        # Add optional fields if provided
        if location:
            event['location'] = location
        if description:
            event['description'] = description
        if attendees:
            event['attendees'] = [{'email': email} for email in attendees]

        # Create the event
        creds = get_creds()
        service = build("calendar", "v3", credentials=creds)
        created_event = service.events().insert(
            calendarId=calendar_id,
            body=event,
            sendNotifications=send_notifications).execute()

        return created_event

    except Exception as e:
        logging.error(f"Error creating calendar event: {str(e)}")
        logging.error(traceback.format_exc())
        return None


@mcp.tool()
def delete_event(event_id: str,
                 send_notifications: bool = True,
                 calendar_id: str = 'primary') -> bool:
    """
    Delete a calendar event by its ID.
    
    Args:
        event_id (str): The ID of the event to delete
        send_notifications (bool): Whether to send cancellation notifications to attendees
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        creds = get_creds()
        service = build("calendar", "v3", credentials=creds)
        service.events().delete(
            calendarId=calendar_id,
            eventId=event_id,
            sendNotifications=send_notifications).execute()
        return True

    except Exception as e:
        logging.error(f"Error deleting calendar event {event_id}: {str(e)}")
        logging.error(traceback.format_exc())
        return False


def get_client():
    user_name = os.getenv('LINKEDIN_NAME')
    user_pass = os.getenv('LINKEDIN_PASSWORD')

    api = Linkedin(user_name, user_pass)
    return api


@mcp.tool()
def get_feeds(public_ids: list[str]) -> list[str]:
    """
    Retrives Recent Linkedin Posts from users.
    
    Args:
        list[str]: list of public_ids of the linkedin users to fetch the feeds.
        
    Returns:
        list[str]: List of recent posts from the linkedin users.
    """

    api = get_client()

    links = []
    posts = []
    for id in public_ids:
        feed_result = api.get_profile_posts(public_id=id, post_count=2)
        author = feed_result[0]['actor']['name']['text']
        for idx, result in enumerate(feed_result):
            text = f"Author:{author}\n\n{result['commentary']['text']['text']}"

            posts.append({"idx": idx, "post": text})
            url_pattern = r"https?://\S+"
            urls = re.findall(url_pattern, text)
            for url in urls:
                links.append({"idx": idx, "url": url})

    return posts


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
