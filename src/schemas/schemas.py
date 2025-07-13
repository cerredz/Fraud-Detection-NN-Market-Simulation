from pydantic import BaseModel


class Customer(BaseModel):
    pass

class FraudCustomer(BaseModel):
    pass

class Transaction(BaseModel):
    pass

class CreditCard(BaseModel):
    pass

class Account(BaseModel):
    pass
    
class Market(BaseModel):
    pass