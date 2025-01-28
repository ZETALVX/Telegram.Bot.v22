using System; 
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Telegram.Bot;
using Telegram.Bot.Exceptions;
using Telegram.Bot.Polling;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;
using Telegram.Bot.Types.ReplyMarkups;
// Tor
using System.Net;
using System.Net.Http;
using Microsoft.Extensions.Configuration;
using System.Reflection.Metadata;

namespace BotTestTelegram
{
    class Program
    {
        // Global variables
        static int blockLevel = 0;
        static bool messDeleted = false;
        static string[] badWords = new string[] { "bad word", "badword" };
        static string[] veryBadWords = new string[] { "very bad word", "verybadword" };
        static int pollMessageId = 0;

        static async Task Main(string[] args)
        {

            // Tor Proxy Example - Use only if you want active the proxy

            // Note that Tor has to be active at all times for the bot to work.

            // Open the torcc file with a text editor (Found in Tor Browser\Browser\TorBrowser\Data\Tor)
            // Add the following lines (this is an example, you can modify the lines):
            // EntryNodes { NL}
            // ExitNodes { NL}
            // StrictNodes 1
            // SocksPort 127.0.0.1:9050

           /* WebProxy proxy = new("socks5://127.0.0.1:9050"); //This line tells tor to listen on port 9050 for any socks connections. You can change the port to anything you want (9050 is just the default), only make sure to use the same port in your code.

            HttpClient httpClient = new(
                new SocketsHttpHandler { Proxy = proxy, UseProxy = true }
            );
            var botClient = new TelegramBotClient("TOKEN", httpClient);*/

            // Bot Token without Tor Proxy
            var botClient = new TelegramBotClient("TOKEN");

            // Cancellation token
            using var cts = new CancellationTokenSource();

            // Get bot information
            var me = await botClient.GetMe();           

            // Write a welcome message to the console
            Console.WriteLine($"\nHello! I'm {me.Username} and I'm your Bot!");

            // Receiver optionsabilire la connessione. Rifiuto persistente del computer di destinazione. (127.0.0.1:9050)
            var receiverOptions = new ReceiverOptions
            {
                AllowedUpdates = { } // Receive all update types
            };

            // Start receiving updates using DefaultUpdateHandler
            botClient.StartReceiving(
                new DefaultUpdateHandler(HandleUpdateAsync, HandleErrorAsync),
                receiverOptions,
                cts.Token
            );

            // Keep the application running until a key is pressed
            Console.ReadKey();

            // Send cancellation request to stop the bot
            cts.Cancel();
        }

        // Method to handle updates
        static async Task HandleUpdateAsync(ITelegramBotClient botClient, Update update, CancellationToken cancellationToken)
        {
            // Check if the update contains a message
            if (update.Message == null)
                return;

            // Check if the message is of type text
            if (update.Message.Type != MessageType.Text)
                return;

            // Set variables
            var chatId = update.Message.Chat.Id;
            var messageText = update.Message.Text.ToLower();
            var messageId = update.Message.MessageId;
            var firstName = update.Message.From?.FirstName ?? "";
            var lastName = update.Message.From?.LastName ?? "";
            var userId = update.Message.From?.Id ?? 0;
            var date = update.Message.Date;

            // Display date and time in console when a message is received
            Console.WriteLine($"\nDate message --> {date:yyyy/MM/dd - HH:mm:ss}");
            // Display the message, chat ID, and user info in console
            Console.WriteLine($"Received a '{messageText}' message in chat {chatId} from user:\n{firstName} - {lastName} - {userId}");

            // Check if the user is a member of the required channels
            try
            {
                var getChatMember1 = await botClient.GetChatMember("@tryerthyhdtyhd", userId);
                var getChatMember2 = await botClient.GetChatMember("@tryerthyhdtyhd", userId);

                // If the user is not a member, prompt them to join
                if (getChatMember1.Status == ChatMemberStatus.Left || getChatMember2.Status == ChatMemberStatus.Left)
                {
                    // Create buttons with the channel URLs to follow
                    InlineKeyboardMarkup inlineKeyboard = new InlineKeyboardMarkup(new[]
                    {
                        new []
                        {
                            InlineKeyboardButton.WithUrl(text: "Channel 1", url: "https://t.me/tryerthyhdtyhd"),
                            InlineKeyboardButton.WithUrl(text: "Channel 2", url: "https://t.me/tryerthyhdtyhd"),
                        },
                    });

                    await botClient.SendMessage(
                        chatId,
                        text: "Before using the bot, you must follow these channels",
                        replyMarkup: inlineKeyboard,
                        cancellationToken: cancellationToken);
                    return; // Exit early since user needs to follow channels
                }
            }
            catch (ApiRequestException ex)
            {
                Console.WriteLine($"Error fetching chat member: {ex.Message}");
                return; // Exit early if unable to fetch chat member
            }

            // Handle the /vulgarity command to change the block level
            if (messageText == "/vulgarity")
            {
                switch (blockLevel)
                {
                    case 0:
                        blockLevel = 1;
                        await botClient.SendMessage(
                            chatId,
                            text: "Vulgarity: \"Medium block\".");
                        return;

                    case 1:
                        blockLevel = 2;
                        await botClient.SendMessage(
                            chatId,
                            text: "Vulgarity: \"Hard block\".");
                        return;

                    case 2:
                        blockLevel = 0;
                        await botClient.SendMessage(
                            chatId,
                            text: "Vulgarity: \"Block disabled\".");
                        return;
                }
            }

            // Vulgarity block - bad words
            foreach (var badWord in badWords)
            {
                if (messageText.Contains(badWord) && blockLevel == 2 && !messDeleted)
                {
                    messDeleted = true;
                    await botClient.DeleteMessage(chatId, messageId, cancellationToken);
                    await botClient.SendMessage(
                        chatId,
                        $"{firstName} {lastName}, you can't say that.");
                }
            }

            // Vulgarity block - very bad words
            foreach (var veryBadWord in veryBadWords)
            {
                if (messageText.Contains(veryBadWord) && (blockLevel == 1 || blockLevel == 2) && !messDeleted)
                {
                    messDeleted = true;
                    await botClient.DeleteMessage(chatId, messageId, cancellationToken);
                    await botClient.SendMessage(
                        chatId,
                        $"{firstName} {lastName}, you can't say that.");
                }
            }
            messDeleted = false;

            // Respond to specific commands
            if (messageText == "hello")
            {
                await botClient.SendMessage(
                    chatId,
                    $"Hello {firstName} {lastName}",
                    cancellationToken: cancellationToken);
            }

            //Sending sticker message
            if (messageText == "sticker")
            {
                await botClient.SendSticker(chatId, "https://telegrambots.github.io/book/docs/sticker-dali.webp");
            }

            // Sending voice
            if (messageText == "voice")
            {
                await using Stream stream = File.OpenRead("/path/to/voice-nfl_commentary.ogg");
                await botClient.SendVoice(chatId, stream, duration: 36);
            }

            // Send a meme photo
            if (messageText == "meme")
            {
                await botClient.SendPhoto(
                    chatId,
                    "https://i.redd.it/uhkj4abc96r61.jpg",
                    caption: "<b>MEME</b>",
                    parseMode: ParseMode.Html);
            }

            // Send audio file
            if (messageText == "sound")
            {
                await botClient.SendAudio(
                    chatId,
                   "https://github.com/TelegramBots/book/raw/master/src/docs/audio-guitar.mp3");
            }

            // Send video
            if (messageText == "video")
            {
                await botClient.SendVideo(
                    chatId,
                    "https://raw.githubusercontent.com/TelegramBots/book/master/src/docs/video-countdown.mp4",
                    thumbnail: "https://telegrambots.github.io/book/2/docs/thumb-clock.jpg",
                    supportsStreaming: true);
            }

            //Send a Video Note
            if (messageText == "video note")
            {
                //Download the Sea Waves video to your disk for this example.
                await using Stream stream = File.OpenRead("/path/to/video-waves.mp4");

                await botClient.SendVideoNote(chatId, stream,
                    duration: 47, length: 360); // value of width/height
            }

            // Send album of photos
            if (messageText == "album")
            {
                await botClient.SendMediaGroup(
                    chatId,
                    new IAlbumInputMedia[]
                    {
                        new InputMediaPhoto("https://cdn.pixabay.com/photo/2017/06/20/19/22/fuchs-2424369_640.jpg"),
                        new InputMediaPhoto("https://cdn.pixabay.com/photo/2017/04/11/21/34/giraffe-2222908_640.jpg"),
                    });
            }

            // Send document
            if (messageText == "doc")
            {
                await botClient.SendDocument(
                    chatId,
                    "https://github.com/TelegramBots/book/raw/master/src/docs/photo-ara.jpg",
                    "<b>Ara bird</b>. <i>Source</i>: <a href=\"https://pixabay.com\">Pixabay</a>",
                    parseMode: ParseMode.Html);
            }

            // Send an animation (GIF) 
            if (messageText == "gif")
            {
                await botClient.SendAnimation(
                    chatId,
                    "https://raw.githubusercontent.com/TelegramBots/book/master/src/docs/video-waves.mp4",
                    "Waves");
            }

            // Create a poll
            if (messageText == "poll")
            {
                var pollMessage = await botClient.SendPoll(
                    chatId,
                    "How are you?",
                    new InputPollOption[]
                    {
                        "Good!",
                        "I could be better.."
                    });

                // Save the poll message ID
                pollMessageId = pollMessage.MessageId;
                Console.WriteLine($"\nPoll number: {pollMessageId}!");
            }

            // Close the poll
            if (messageText == "close poll" && pollMessageId != 0)
            {
                await botClient.StopPoll(
                    chatId,
                    pollMessageId);

                Console.WriteLine($"\nPoll number {pollMessageId} is closed!");
            }

            // Send a contact
            if (messageText == "contact")
            {
                await botClient.SendContact(
                    chatId,
                    phoneNumber: "+1234567890",
                    firstName: "Han",
                    lastName: "Solo",
                    vcard: "BEGIN:VCARD\n" +
                   "VERSION:3.0\n" +
                   "N:Solo;Han\n" +
                   "ORG:Scruffy-looking nerf herder\n" +
                   "TEL;TYPE=voice,work,pref:+1234567890\n" +
                   "EMAIL:hansolo@mfalcon.com\n" +
                   "END:VCARD");
            }

            // Send a venue
            if (messageText == "roma location")
            {
                await botClient.SendVenue(
                    chatId,
                    latitude: 41.9027835f,
                    longitude: 12.4963655f,
                    title: "Rome",
                    address: "Rome, via Daqua 8, 08089");
            }

            // Send a location
            if (messageText == "send me a location")
            {
                await botClient.SendLocation(
                    chatId: chatId,
                    latitude: 41.9027835f,
                    longitude: 12.4963655f);
            }
        }

        // Method to handle errors
        static Task HandleErrorAsync(ITelegramBotClient botClient, Exception exception, CancellationToken cancellationToken)
        {
            // Error message based on exception type
            var errorMessage = exception switch
            {
                ApiRequestException apiRequestException
                    => $"Telegram API Error:\n[{apiRequestException.ErrorCode}]\n{apiRequestException.Message}",
                _ => exception.ToString()
            };

            // Print error message to console
            Console.WriteLine(errorMessage);
            return Task.CompletedTask;
        }
    }
}
