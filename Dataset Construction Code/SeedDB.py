from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timezone
import random

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["ecommerce_demo"]

# Drop existing data
db.login.drop()
db.users.drop()
db.products.drop()
db.orders.drop()

# 1. Create Users
users = []
for i in range(5):
    user_id = ObjectId()
    users.append({
        "_id": user_id,
        "name": f"User {i+1}",
        "email": f"user{i+1}@example.com",
        "phone": f"+12345678{i+1}",
        "address": {
            "street": f"{100+i} Demo Street",
            "city": "Demo City",
            "zip": f"1000{i}",
            "country": "USA"
        },
        "is_admin": (i == 0),  # First user is admin
        "status": "active" if i % 2 == 0 else "inactive",
        "created_at": datetime.now(timezone.utc)
    })

db.users.insert_many(users)

# 2. Create Logins
login = []
for i, user in enumerate(users):
    login.append({
        "user_id": user["_id"],
        "username": f"user{i+1}",
        "password": f"$2b$10$hash{i+1}",  
        "role": "admin" if user["is_admin"] else "user"
    })

db.login.insert_many(login)

# 3. Create Products
products = []
for i in range(5):
    product_id = ObjectId()
    products.append({
        "_id": product_id,
        "name": f"Product {i+1}",
        "description": f"This is product {i+1}",
        "price": round(random.uniform(10, 100), 2),
        "stock": random.randint(10, 100),
        "availability": random.choice([True, False]),
        "category": "General",
        "created_at": datetime.now(timezone.utc)
    })

db.products.insert_many(products)

# 4. Create Orders
orders = []
for i in range(5):
    user = random.choice(users)
    selected_products = random.sample(products, k=2)
    items = []
    total = 0

    for product in selected_products:
        quantity = random.randint(1, 3)
        items.append({
            "product_id": product["_id"],
            "quantity": quantity,
            "price": product["price"]
        })
        total += quantity * product["price"]

    order_id = ObjectId()
    order_date = datetime.now(timezone.utc)
    refund_requested = random.choice([True, False])
    refund_amount = round(random.uniform(1, total), 2) if refund_requested else 0.0

    orders.append({
        "_id": order_id,
        "user_id": user["_id"],
        "items": items,
        "total": round(total, 2),
        "status": random.choice(["processing", "shipped", "delivered"]),
        "shipping_address": user["address"],
        "order_date": order_date,
        "payment_method": random.choice(["credit_card", "paypal", "bank_transfer"]),
        "payment_status": "paid",
        "paid_at": datetime.now(timezone.utc),
        "transaction_id": ObjectId(),
        "refund_requested": refund_requested,
        "refund_amount": refund_amount
    })

db.orders.insert_many(orders)

print("Database seeded successfully with updated fields and 5 records per collection.")
