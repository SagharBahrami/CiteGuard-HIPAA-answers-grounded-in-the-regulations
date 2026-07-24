"""Hand-curated eval cases for the faithfulness guardrail.

Each case pairs a query with real retrieved chunks (captured once via
retriever.retrieve and frozen here as plain dicts, so eval results aren't
confounded by retrieval drift -- this eval isolates check_faithfulness
specifically, not the whole pipeline) and an answer that is either
genuinely grounded in those chunks or contains a deliberately fabricated
claim not supported by them. Faithful answers were checked by hand against
the chunk text below, not generated live, so the labels are trustworthy
ground truth rather than a re-run of the thing being evaluated.
"""

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    name: str
    query: str
    chunks: list[dict]
    answer: str
    expected_faithful: bool


_ENCRYPTION_CHUNKS = [
    {
        "citation": "45 CFR 164.312",
        "heading": "§ 164.312 Technical safeguards.",
        "part": 164,
        "subpart": "Subpart C—Security Standards for the Protection of Electronic Protected Health Information",
        "text": (
            "(c)(1) Standard: Integrity. Implement policies and procedures to protect electronic protected "
            "health information from improper alteration or destruction.\n"
            "(2) Implementation specification: Mechanism to authenticate electronic protected health "
            "information (Addressable). Implement electronic mechanisms to corroborate that electronic "
            "protected health information has not been altered or destroyed in an unauthorized manner.\n"
            "(d) Standard: Person or entity authentication. Implement procedures to verify that a person or "
            "entity seeking access to electronic protected health information is the one claimed.\n"
            "(e)(1) Standard: Transmission security. Implement technical security measures to guard against "
            "unauthorized access to electronic protected health information that is being transmitted over an "
            "electronic communications network.\n"
            "(2) Implementation specifications:\n"
            "(i) Integrity controls (Addressable). Implement security measures to ensure that electronically "
            "transmitted electronic protected health information is not improperly modified without detection "
            "until disposed of.\n"
            "(ii) Encryption (Addressable). Implement a mechanism to encrypt electronic protected health "
            "information whenever deemed appropriate."
        ),
        "similarity": 0.5149605332900754,
    },
    {
        "citation": "45 CFR 164.304",
        "heading": "§ 164.304 Definitions.",
        "part": 164,
        "subpart": "Subpart C—Security Standards for the Protection of Electronic Protected Health Information",
        "text": (
            "Integrity means the property that data or information have not been altered or destroyed in an "
            "unauthorized manner.\nMalicious software means software, for example, a virus, designed to damage "
            "or disrupt a system.\nPassword means confidential authentication information composed of a string "
            "of characters."
        ),
        "similarity": 0.47769831878342073,
    },
    {
        "citation": "45 CFR 164.310",
        "heading": "§ 164.310 Physical safeguards.",
        "part": 164,
        "subpart": "Subpart C—Security Standards for the Protection of Electronic Protected Health Information",
        "text": (
            "(b) Standard: Workstation use. Implement policies and procedures that specify the proper functions "
            "to be performed, the manner in which those functions are to be performed, and the physical "
            "attributes of the surroundings of a specific workstation or class of workstation that can access "
            "electronic protected health information.\n"
            "(c) Standard: Workstation security. Implement physical safeguards for all workstations that access "
            "electronic protected health information, to restrict access to authorized users."
        ),
        "similarity": 0.47304777334660675,
    },
]

_BA_CHUNKS = [
    {
        "citation": "45 CFR 160.103",
        "heading": "§ 160.103 Definitions.",
        "part": 160,
        "subpart": "Subpart A—General Provisions",
        "text": (
            "(3) Business associate includes:\n"
            "(i) A Health Information Organization, E-prescribing Gateway, or other person that provides data "
            "transmission services with respect to protected health information to a covered entity and that "
            "requires access on a routine basis to such protected health information.\n"
            "(ii) A person that offers a personal health record to one or more individuals on behalf of a "
            "covered entity.\n"
            "(iii) A subcontractor that creates, receives, maintains, or transmits protected health information "
            "on behalf of the business associate."
        ),
        "similarity": 0.6201314186987944,
    },
    {
        "citation": "45 CFR 164.504",
        "heading": "§ 164.504 Uses and disclosures: Organizational requirements.",
        "part": 164,
        "subpart": "Subpart E—Privacy of Individually Identifiable Health Information",
        "text": (
            "(e)(1) Standard: Business associate contracts. (i) The contract or other arrangement required by "
            "§ 164.502(e)(2) must meet the requirements of paragraph (e)(2), (e)(3), or (e)(5) of this section, "
            "as applicable.\n"
            "(ii) A covered entity is not in compliance with the standards in § 164.502(e) and this paragraph, "
            "if the covered entity knew of a pattern of activity or practice of the business associate that "
            "constituted a material breach or violation of the business associate's obligation under the "
            "contract or other arrangement, unless the covered entity took reasonable steps to cure the breach "
            "or end the violation, as applicable, and, if such steps were unsuccessful, terminated the contract "
            "or arrangement, if feasible."
        ),
        "similarity": 0.5879858042335419,
    },
    {
        "citation": "45 CFR 164.314",
        "heading": "§ 164.314 Organizational requirements.",
        "part": 164,
        "subpart": "Subpart C—Security Standards for the Protection of Electronic Protected Health Information",
        "text": (
            "(a)(2) Implementation specifications (Required)—(i) Business associate contracts. The contract "
            "must provide that the business associate will—\n"
            "(A) Comply with the applicable requirements of this subpart;\n"
            "(B) In accordance with § 164.308(b)(2), ensure that any subcontractors that create, receive, "
            "maintain, or transmit electronic protected health information on behalf of the business associate "
            "agree to comply with the applicable requirements of this subpart by entering into a contract or "
            "other arrangement that complies with this section; and\n"
            "(C) Report to the covered entity any security incident of which it becomes aware, including "
            "breaches of unsecured protected health information as required by § 164.410."
        ),
        "similarity": 0.5834217378545945,
    },
]

_BREACH_CHUNKS = [
    {
        "citation": "45 CFR 164.406",
        "heading": "§ 164.406 Notification to the media.",
        "part": 164,
        "subpart": "Subpart D—Notification in the Case of Breach of Unsecured Protected Health Information",
        "text": (
            "(a) Standard. For a breach of unsecured protected health information involving more than 500 "
            "residents of a State or jurisdiction, a covered entity shall, following the discovery of the "
            "breach as provided in § 164.404(a)(2), notify prominent media outlets serving the State or "
            "jurisdiction.\n"
            "(b) Implementation specification: Timeliness of notification. Except as provided in § 164.412, a "
            "covered entity shall provide the notification required by paragraph (a) of this section without "
            "unreasonable delay and in no case later than 60 calendar days after discovery of a breach."
        ),
        "similarity": 0.6995045149476344,
    },
    {
        "citation": "45 CFR 164.404",
        "heading": "§ 164.404 Notification to individuals.",
        "part": 164,
        "subpart": "Subpart D—Notification in the Case of Breach of Unsecured Protected Health Information",
        "text": (
            "(c) Implementation specifications: Content of notification—(1) Elements. The notification required "
            "by paragraph (a) of this section shall include, to the extent possible:\n"
            "(A) A brief description of what happened, including the date of the breach and the date of the "
            "discovery of the breach, if known;\n"
            "(B) A description of the types of unsecured protected health information that were involved in "
            "the breach (such as whether full name, social security number, date of birth, home address, "
            "account number, diagnosis, disability code, or other types of information were involved);"
        ),
        "similarity": 0.6790249966295685,
    },
    {
        "citation": "45 CFR 164.410",
        "heading": "§ 164.410 Notification by a business associate.",
        "part": 164,
        "subpart": "Subpart D—Notification in the Case of Breach of Unsecured Protected Health Information",
        "text": (
            "(b) Implementation specifications: Timeliness of notification. Except as provided in § 164.412, a "
            "business associate shall provide the notification required by paragraph (a) of this section "
            "without unreasonable delay and in no case later than 60 calendar days after discovery of a breach."
        ),
        "similarity": 0.672206527151924,
    },
]

_MIN_NECESSARY_CHUNKS = [
    {
        "citation": "45 CFR 164.514",
        "heading": "§ 164.514 Other requirements relating to uses and disclosures of protected health information.",
        "part": 164,
        "subpart": "Subpart E—Privacy of Individually Identifiable Health Information",
        "text": (
            "(iii) A covered entity may rely, if such reliance is reasonable under the circumstances, on a "
            "requested disclosure as the minimum necessary for the stated purpose when:\n"
            "(A) Making disclosures to public officials that are permitted under § 164.512, if the public "
            "official represents that the information requested is the minimum necessary for the stated "
            "purpose(s);\n"
            "(B) The information is requested by another covered entity;\n"
            "(C) The information is requested by a professional who is a member of its workforce or is a "
            "business associate of the covered entity for the purpose of providing professional services to "
            "the covered entity, if the professional represents that the information requested is the minimum "
            "necessary for the stated purpose(s).\n"
            "(4) Implementation specifications: Minimum necessary requests for protected health information. "
            "(i) A covered entity must limit any request for protected health information to that which is "
            "reasonably necessary to accomplish the purpose for which the request is made, when requesting "
            "such information from other covered entities."
        ),
        "similarity": 0.5039065056572087,
    },
    {
        "citation": "45 CFR 164.502",
        "heading": "§ 164.502 Uses and disclosures of protected health information: General rules.",
        "part": 164,
        "subpart": "Subpart E—Privacy of Individually Identifiable Health Information",
        "text": (
            "(b) Standard: Minimum necessary— Minimum necessary applies. When using or disclosing protected "
            "health information or when requesting protected health information from another covered entity or "
            "business associate, a covered entity or business associate must make reasonable efforts to limit "
            "protected health information to the minimum necessary to accomplish the intended purpose of the "
            "use, disclosure, or request.\n"
            "(2) Minimum necessary does not apply. This requirement does not apply to:\n"
            "(i) Disclosures to or requests by a health care provider for treatment;"
        ),
        "similarity": 0.46261745966286805,
    },
]

_WORKFORCE_CHUNKS = [
    {
        "citation": "45 CFR 164.308",
        "heading": "§ 164.308 Administrative safeguards.",
        "part": 164,
        "subpart": "Subpart C—Security Standards for the Protection of Electronic Protected Health Information",
        "text": (
            "(2) Standard: Assigned security responsibility. Identify the security official who is responsible "
            "for the development and implementation of the policies and procedures required by this subpart "
            "for the covered entity or business associate.\n"
            "(3)(i) Standard: Workforce security. Implement policies and procedures to ensure that all members "
            "of its workforce have appropriate access to electronic protected health information, as provided "
            "under paragraph (a)(4) of this section, and to prevent those workforce members who do not have "
            "access under paragraph (a)(4) of this section from obtaining access to electronic protected "
            "health information.\n"
            "(ii) Implementation specifications:\n"
            "(A) Authorization and/or supervision (Addressable). Implement procedures for the authorization "
            "and/or supervision of workforce members who work with electronic protected health information or "
            "in locations where it might be accessed.\n"
            "(B) Workforce clearance procedure (Addressable). Implement procedures to determine that the "
            "access of a workforce member to electronic protected health information is appropriate.\n"
            "(C) Termination procedures (Addressable). Implement procedures for terminating access to "
            "electronic protected health information when the employment of, or other arrangement with, a "
            "workforce member ends."
        ),
        "similarity": 0.6049228884560761,
    },
    {
        "citation": "45 CFR 164.530",
        "heading": "§ 164.530 Administrative requirements.",
        "part": 164,
        "subpart": "Subpart E—Privacy of Individually Identifiable Health Information",
        "text": (
            "(B) Thereafter, to each new member of the workforce within a reasonable period of time after the "
            "person joins the covered entity's workforce; and\n"
            "(C) To each member of the covered entity's workforce whose functions are affected by a material "
            "change in the policies or procedures required by this subpart or subpart D of this part, within a "
            "reasonable period of time after the material change becomes effective.\n"
            "(ii) A covered entity must document that the training as described in paragraph (b)(2)(i) of this "
            "section has been provided, as required by paragraph (j) of this section."
        ),
        "similarity": 0.5407003680290381,
    },
]


CASES: list[EvalCase] = [
    EvalCase(
        name="encryption-faithful",
        query="What are the technical safeguards for encryption?",
        chunks=_ENCRYPTION_CHUNKS,
        answer=(
            "Under 45 CFR 164.312(e)(1), covered entities must implement technical security measures to guard "
            "against unauthorized access to electronic protected health information (ePHI) transmitted over an "
            "electronic communications network (the Transmission security standard). Under 45 CFR "
            "164.312(e)(2)(ii), encryption is an addressable implementation specification: covered entities "
            "must implement a mechanism to encrypt ePHI whenever deemed appropriate. Related integrity "
            "mechanisms are addressed under 45 CFR 164.312(c), which requires an addressable mechanism to "
            "authenticate that ePHI has not been altered or destroyed in an unauthorized manner."
        ),
        expected_faithful=True,
    ),
    EvalCase(
        name="encryption-unfaithful",
        query="What are the technical safeguards for encryption?",
        chunks=_ENCRYPTION_CHUNKS,
        answer=(
            "Under 45 CFR 164.312, covered entities must encrypt all electronic protected health information "
            "using AES-256 encryption both at rest and in transit, as a required (not addressable) "
            "specification. Additionally, 45 CFR 164.304 defines 'Integrity' as requiring cryptographic "
            "checksums to be recalculated every 24 hours."
        ),
        expected_faithful=False,
    ),
    EvalCase(
        name="business-associate-faithful",
        query="What is a business associate agreement?",
        chunks=_BA_CHUNKS,
        answer=(
            "A business associate agreement is the contract required between a covered entity and a business "
            "associate. Under 45 CFR 160.103, a business associate includes entities such as a Health "
            "Information Organization or E-prescribing Gateway providing data transmission services requiring "
            "routine access to protected health information, a person offering a personal health record on "
            "behalf of a covered entity, and subcontractors that create, receive, maintain, or transmit "
            "protected health information on behalf of the business associate. Under 45 CFR 164.504(e), a "
            "covered entity is not in compliance if it knew of a pattern of activity or practice by the "
            "business associate that constituted a material breach of the business associate's contractual "
            "obligations, unless the covered entity took reasonable steps to cure the breach. Under 45 CFR "
            "164.314(a), the contract must require the business associate to comply with applicable subpart "
            "requirements, ensure subcontractors handling ePHI agree to the same restrictions, and report any "
            "security incident, including breaches, to the covered entity."
        ),
        expected_faithful=True,
    ),
    EvalCase(
        name="business-associate-unfaithful",
        query="What is a business associate agreement?",
        chunks=_BA_CHUNKS,
        answer=(
            "A business associate agreement, under 45 CFR 164.504(e), must be renewed annually and requires "
            "the business associate to carry a minimum of $2 million in cyber-liability insurance. Under 45 "
            "CFR 164.314(a), the contract automatically terminates after 3 years unless renegotiated."
        ),
        expected_faithful=False,
    ),
    EvalCase(
        name="breach-notification-faithful",
        query="What are the breach notification requirements?",
        chunks=_BREACH_CHUNKS,
        answer=(
            "Under 45 CFR 164.406, if a breach of unsecured protected health information involves more than "
            "500 residents of a State or jurisdiction, the covered entity must notify prominent media outlets "
            "serving that State or jurisdiction, without unreasonable delay and no later than 60 calendar days "
            "after discovery. Under 45 CFR 164.404, notification to affected individuals must include a brief "
            "description of what happened (including breach and discovery dates, if known) and a description "
            "of the types of unsecured protected health information involved. Under 45 CFR 164.410, a business "
            "associate must similarly provide notification without unreasonable delay and no later than 60 "
            "calendar days after discovery of a breach."
        ),
        expected_faithful=True,
    ),
    EvalCase(
        name="breach-notification-unfaithful",
        query="What are the breach notification requirements?",
        chunks=_BREACH_CHUNKS,
        answer=(
            "Under 45 CFR 164.406, any breach affecting even a single individual requires immediate "
            "notification to national media within 24 hours. Under 45 CFR 164.410, business associates have "
            "120 days to notify affected individuals, twice the deadline given to covered entities."
        ),
        expected_faithful=False,
    ),
    EvalCase(
        name="minimum-necessary-faithful",
        query="What is the minimum necessary standard?",
        chunks=_MIN_NECESSARY_CHUNKS,
        answer=(
            "The minimum necessary standard, under 45 CFR 164.502(b), requires that when using, disclosing, or "
            "requesting protected health information, a covered entity or business associate must make "
            "reasonable efforts to limit the information to the minimum necessary to accomplish the intended "
            "purpose. This requirement does not apply to disclosures to or requests by a health care provider "
            "for treatment. Under 45 CFR 164.514, a covered entity may reasonably rely on a requested "
            "disclosure as meeting the minimum necessary standard in certain circumstances — for example, when "
            "requested by another covered entity, by a workforce member or business associate providing "
            "professional services, or when a public official represents the request is minimum necessary "
            "under 45 CFR 164.512."
        ),
        expected_faithful=True,
    ),
    EvalCase(
        name="minimum-necessary-unfaithful",
        query="What is the minimum necessary standard?",
        chunks=_MIN_NECESSARY_CHUNKS,
        answer=(
            "The minimum necessary standard under 45 CFR 164.514 requires covered entities to disclose no more "
            "than 3 specific data elements per request, and mandates that a written minimum-necessary "
            "determination form be filed with HHS within 10 days of any disclosure. Under 45 CFR 164.502(b), "
            "this standard applies uniformly to all disclosures without exception, including treatment-related "
            "requests."
        ),
        expected_faithful=False,
    ),
    EvalCase(
        name="workforce-security-faithful",
        query="What are the administrative safeguards for workforce security?",
        chunks=_WORKFORCE_CHUNKS,
        answer=(
            "Under 45 CFR 164.308, a covered entity or business associate must identify a security official "
            "responsible for developing and implementing the required security policies and procedures (the "
            "'Assigned security responsibility' standard), and must implement policies and procedures to "
            "ensure that workforce members have appropriate access to electronic protected health information "
            "while preventing members without authorized access from obtaining it (the 'Workforce security' "
            "standard) — including addressable specifications for authorization/supervision, a workforce "
            "clearance procedure, and termination procedures. Under 45 CFR 164.530, covered entities must "
            "provide training to new workforce members within a reasonable time after they join, and to "
            "existing members when policies or procedures materially change, and must document that this "
            "training was provided."
        ),
        expected_faithful=True,
    ),
    EvalCase(
        name="workforce-security-unfaithful",
        query="What are the administrative safeguards for workforce security?",
        chunks=_WORKFORCE_CHUNKS,
        answer=(
            "Under 45 CFR 164.308, covered entities must conduct FBI background checks on all workforce "
            "members before granting any system access, and must terminate access within 1 hour of an "
            "employee's departure. 45 CFR 164.530 requires annual workforce security training with a passing "
            "exam score of at least 80%, documented and submitted to the Office for Civil Rights."
        ),
        expected_faithful=False,
    ),
]
